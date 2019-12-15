import tflearn
import numpy as np
import tensorflow as tf
import time
from arguments import get_args
args = get_args()

ENTROPY_WEIGHT = args.entropy_coef
ENTROPY_EPS = 1e-6
FC0_OUT = 128
CONV1D_OUTCHANNEL = 4
GAMMA = args.gamma

FIRST_LAYER = 1024
SECOND_LAYER = 512

C_FIRST_LAYER = 400
C_SECOND_LAYER = 150

class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.entropy_weigh = ENTROPY_WEIGHT
        self.startTime = time.time()

        # Create the actor network
        self.inputs, self.out, self.dense_net_0 = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(tf.multiply(
            tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
                                 reduction_indices=1, keep_dims=True)),
            -self.act_grad_weights)) \
                   + self.entropy_weigh * tf.reduce_sum(tf.multiply(self.out,
                                                                tf.log(self.out + ENTROPY_EPS)))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            inputs = tflearn.input_data(shape=[None, self.s_dim])

            # split_0: 0~7: indicates the measured available throughput 8 -> FC_OUT
            # split_1: 8: the sum of downloaded bitrate of chunks 1 -> FC_OUT
            # split_2: 9~10: the difference of QoE among videos 2 -> FC_OUT
            # split_3: 11~12: the number of clients who request this video, one dim for phone, one for tv, 2 -> FC_OUT
            # split_4: 13~22: predicted ph VMAF, 10 -> FC_OUT
            # split_5: 23~32: predicted tv VMAF, 10 -> FC_OUT
            # split_6: 33~34: the VMAF of last requested chunk, for ph and tv respectively, 2 -> FC_OUT
            # split_7: 35: the download rate of last chunk, 1 -> FC_OUT
            # split_8: 36: the download time of last chunk, 1 -> FC_OUT
            # split_9: 37~38: indicates the chunk skip events, 00:no skip, 01:skip 1~2, 11:skip over 3, 2 -> FC_OUT
            # split_10: 39: How long the chunk remain in CDN. 1 -> FC_OUT
            # split_11: -1: indicates the delay of the video, 1 -> FC_OUT

            split0_inputs = tf.reshape(inputs[:, :8], [-1, 1, 8])
            split1_inputs = tf.reshape(inputs[:, 8], [-1, 1])
            split2_inputs = tf.reshape(inputs[:, 9:11], [-1, 2])
            split3_inputs = tf.reshape(inputs[:, 11:13], [-1, 2])
            split4_inputs = tf.reshape(inputs[:, 13:23], [-1, 10])
            split5_inputs = tf.reshape(inputs[:, 23:33], [-1, 10])
            split6_inputs = tf.reshape(inputs[:, 33:35], [-1, 2])
            split7_inputs = tf.reshape(inputs[:, 35], [-1, 1])
            split8_inputs = tf.reshape(inputs[:, 36], [-1, 1])
            split9_inputs = tf.reshape(inputs[:, 37:39], [-1, 2])
            # split10_inputs = tf.reshape(inputs[:, 39], [-1, 1])
            split11_inputs = tf.reshape(inputs[:, -1], [-1, 1])

            split_0 = tflearn.conv_1d(split0_inputs, FC0_OUT, CONV1D_OUTCHANNEL,
                                      weights_init='xavier', activation='relu')
            split_1 = tflearn.fully_connected(split1_inputs, FC0_OUT,
                                              weights_init='xavier', activation='relu')
            split_2 = tflearn.fully_connected(split2_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_3 = tflearn.fully_connected(split3_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_4 = tflearn.fully_connected(split4_inputs, FC0_OUT, weights_init='variance_scaling', activation='elu')
            split_5 = tflearn.fully_connected(split5_inputs, FC0_OUT, weights_init='variance_scaling', activation='elu')
            split_6 = tflearn.fully_connected(split6_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_7 = tflearn.fully_connected(split7_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_8 = tflearn.fully_connected(split8_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_9 = tflearn.fully_connected(split9_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            # split_10 = tflearn.fully_connected(split10_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_11 = tflearn.fully_connected(split11_inputs, FC0_OUT, weights_init='xavier', activation='relu')

            split_0_flat = tflearn.flatten(split_0)
            merge_net = tflearn.merge([split_0_flat, split_1, split_2, split_3, split_4, split_5,
                                      split_6, split_7, split_8, split_9, split_11], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, FIRST_LAYER, weights_init='xavier', activation='relu')
            dense_net_1 = tflearn.fully_connected(dense_net_0, SECOND_LAYER, weights_init='xavier', activation='relu')

            out = tflearn.fully_connected(dense_net_1, self.a_dim, weights_init='xavier', activation='softmax')

            return inputs, out, dense_net_0

    def train(self, inputs, acts, act_grad_weights):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def print_dense0(self, inputs):
        return self.sess.run(self.split0_inputs, feed_dict={
            self.inputs: inputs
        })

    def save_variable_summaries(self, inputs, step):
        dense0 = self.sess.run(self.dense_net_0, feed_dict={
            self.inputs: inputs
        })
        dense1 = self.sess.run(self.dense_net_1, feed_dict={
            self.inputs: inputs
        })
        tf.summary.histogram('actor' + '/dense_net_0', dense0)
        merge_op = tf.summary.merge_all()
        summary_ops = self.sess.run(merge_op, dense0)
        self.writer.add_summary(summary_ops, step)

    def get_gradients(self, inputs, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        for i, g in enumerate(actor_gradients):
            if g is not None:
                g[np.isnan(g)] = 0
                actor_gradients[i] = g

        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    # Re-config the params
    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate). \
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            inputs = tflearn.input_data(shape=[None, self.s_dim])

            # split_0: 0~7: indicates the measured available throughput 8 -> FC_OUT
            # split_1: 8: the sum of downloaded bitrate of chunks 1 -> FC_OUT
            # split_2: 9~10: the difference of QoE among videos 2 -> FC_OUT
            # split_3: 11~12: the number of clients who request this video, one dim for phone, one for tv, 2 -> FC_OUT
            # split_4: 13~22: predicted ph VMAF, 10 -> FC_OUT
            # split_5: 23~32: predicted tv VMAF, 10 -> FC_OUT
            # split_6: 33~34: the VMAF of last requested chunk, for ph and tv respectively, 2 -> FC_OUT
            # split_7: 35: the download rate of last chunk, 1 -> FC_OUT
            # split_8: 36: the download time of last chunk, 1 -> FC_OUT
            # split_9: 37~38: indicates the chunk skip events, 00:no skip, 01:skip 1~2, 11:skip over 3, 2 -> FC_OUT
            # split_10: 39: How long the chunk remain in CDN. 1 -> FC_OUT
            # split_11: -1: indicates the delay of the video, 1 -> FC_OUT

            split0_inputs = tf.reshape(inputs[:, :8], [-1, 1, 8])
            split1_inputs = tf.reshape(inputs[:, 8], [-1, 1])
            split2_inputs = tf.reshape(inputs[:, 9:11], [-1, 2])
            split3_inputs = tf.reshape(inputs[:, 11:13], [-1, 2])
            split4_inputs = tf.reshape(inputs[:, 13:23], [-1, 10])
            split5_inputs = tf.reshape(inputs[:, 23:33], [-1, 10])
            split6_inputs = tf.reshape(inputs[:, 33:35], [-1, 2])
            split7_inputs = tf.reshape(inputs[:, 35], [-1, 1])
            split8_inputs = tf.reshape(inputs[:, 36], [-1, 1])
            split9_inputs = tf.reshape(inputs[:, 37:39], [-1, 2])
            # split10_inputs = tf.reshape(inputs[:, 39], [-1, 1])
            split11_inputs = tf.reshape(inputs[:, -1], [-1, 1])

            split_0 = tflearn.conv_1d(split0_inputs, FC0_OUT, CONV1D_OUTCHANNEL,
                                      weights_init='xavier', activation='relu')
            split_1 = tflearn.fully_connected(split1_inputs, FC0_OUT,
                                              weights_init='xavier', activation='relu')
            split_2 = tflearn.fully_connected(split2_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_3 = tflearn.fully_connected(split3_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_4 = tflearn.fully_connected(split4_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_5 = tflearn.fully_connected(split5_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_6 = tflearn.fully_connected(split6_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_7 = tflearn.fully_connected(split7_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_8 = tflearn.fully_connected(split8_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_9 = tflearn.fully_connected(split9_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            # split_10 = tflearn.fully_connected(split10_inputs, FC0_OUT, weights_init='xavier', activation='relu')
            split_11 = tflearn.fully_connected(split11_inputs, FC0_OUT, weights_init='xavier', activation='relu')

            split_0_flat = tflearn.flatten(split_0)
            merge_net = tflearn.merge([split_0_flat, split_1, split_2, split_3, split_4, split_5,
                                       split_6, split_7, split_8, split_9, split_11], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, C_FIRST_LAYER, weights_init='xavier', activation='relu')
            dense_net_1 = tflearn.fully_connected(dense_net_0, C_SECOND_LAYER, weights_init='xavier', activation='relu')

            out = tflearn.fully_connected(dense_net_1, 1, weights_init='xavier', activation='linear')

            return inputs, out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        for i, g in enumerate(critic_gradients):
            if g is not None:
                g[np.isnan(g)] = 0
                critic_gradients[i] = g
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    """
        batch of s, a, r is from samples in a sequence
        the format is in np.array([batch_size, s/a/r_dim])
        terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]
    v_batch = critic.predict(s_batch)
    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch
    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)
    return actor_gradients, critic_gradients, td_batch

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    return out

def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H

def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

