import model
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from arguments import get_args
import time
import os

DEFAULT_ARGS = get_args()

def central_agent(net_params_queues, exp_queues, state_space_len, action_space_len, online_id_vector,
                  online_num_agents, lock, args=DEFAULT_ARGS):
    assert len(net_params_queues) == args.total_agents
    assert len(exp_queues) == args.total_agents

    with tf.Session() as sess:
        actor = model.ActorNetwork(sess,
                                   state_dim=state_space_len, action_dim=action_space_len,
                                   learning_rate=args.actor_lr)
        critic = model.CriticNetwork(sess,
                                     state_dim=state_space_len,
                                     learning_rate=args.critic_lr)

        summary_ops, summary_vars = model.build_summaries()

        sess.run(tf.global_variables_initializer())

        if not os.path.exists(os.path.join(args.summary_dir, 'summary')):
            os.mkdir(os.path.join(args.summary_dir, 'summary'))
        writer = tf.summary.FileWriter(os.path.join(args.summary_dir, 'summary'), sess.graph)
        saver = tf.train.Saver(max_to_keep=200)

        # Restore neural net parameters
        nn_model = args.nn_model
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")
        epoch = 0

        # assemble experiences from agents, compute the gradients
        max_reward = -1000000
        total_avg_reward = 0
        total_avg_reward_num = 0
        stime = time.time()
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()

            for i in [i for i, e in enumerate(net_params_queues) if e.empty()]:
                net_params_queues[i].put([actor_net_params, critic_net_params])

            # Record average reward and td loss change in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []
            with lock:
                NotEmptyList = [i for i, e in enumerate(exp_queues) if not e.empty()]

            for i in NotEmptyList:
                with lock:
                    s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    model.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])


            assert len(actor_gradient_batch) == len(critic_gradient_batch)

            if total_agents != 0:
                total_avg_reward += total_reward / total_agents
                total_avg_reward_num += 1
                etime = time.time()
                if (etime - stime) >= (60 * 3):
                    if total_avg_reward_num != 0 and (total_avg_reward / total_avg_reward_num) > max_reward:
                        max_reward = total_avg_reward / total_avg_reward_num
                        save_path = saver.save(sess, args.summary_dir + "/nn_model_ep_best.ckpt")
                        os.system('cp %s/nn_model_ep_best.ckpt.* %s/best/' % (args.summary_dir, args.summary_dir))
                    stime = etime
                    total_avg_reward = 0
                    total_avg_reward_num = 0


            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

                if actor.entropy_weigh - 0.0005 > 0.1:
                    actor.entropy_weigh = actor.entropy_weigh - 0.0005
                    print 'actor entropy_weigh is:%s' % str(actor.entropy_weigh)
                else:
                    actor.entropy_weigh = 0.1

                # log training information
                epoch += 1

                avg_reward = total_reward / total_agents
                avg_td_loss = total_td_loss / total_batch_len
                avg_entropy = total_entropy / total_batch_len

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: avg_td_loss,
                    summary_vars[1]: avg_reward,
                    summary_vars[2]: avg_entropy,
                })

                writer.add_summary(summary_str, epoch)
                writer.flush()

                if epoch % args.model_save_interval == 0:
                    # Save the neural net parameters to disk.
                    save_path = saver.save(sess, args.summary_dir + "/nn_model_ep_" +
                                           str(epoch) + ".ckpt")


class A3C():
    def __init__(self, video_name, state_dim, action_dim, net_params_queue, exp_queue, lock, args=DEFAULT_ARGS):

        self.agent_id = video_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = 0
        self.args = args
        self.net_params_queue = net_params_queue
        self.exp_queue = exp_queue
        self.sess = tf.Session()
        self.lock = lock

        self.s_batch = []
        self.a_batch = []
        self.r_batch = []
        self.entropy_record = []
        self.actor = model.ActorNetwork(self.sess,
                                        state_dim=self.state_dim, action_dim=self.action_dim,
                                        learning_rate=args.actor_lr)

        self.critic = model.CriticNetwork(self.sess,
                                          state_dim=self.state_dim,
                                          learning_rate=args.critic_lr)

        # Initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        self.actor.set_network_params(actor_net_params)
        self.critic.set_network_params(critic_net_params)

    def action(self, state):

        action_prob = self.actor.predict(np.reshape(state, (1, self.state_dim)))
        print '%s action_prob:%s' % (self.agent_id, str(action_prob))

        action_cumsum = np.cumsum(action_prob)
        action = (action_cumsum > np.random.randint(1, self.args.rand_range) / float(self.args.rand_range)).argmax()
        action_vec = np.zeros(self.action_dim)
        action_vec[action] = 1

        self.s_batch.append(np.array(state))
        self.a_batch.append(action_vec)
        self.entropy_record.append(model.compute_entropy(action_prob[0]))

        return action

    def save_reward(self, reward):
        self.r_batch.append(reward)

    def train(self, done):
        assert len(self.s_batch) == len(self.a_batch) == len(self.entropy_record) == len(self.entropy_record)

        # report experience to the coordinator
        if len(self.r_batch) >= self.args.num_steps or done:
            if len(self.s_batch) != 0:
                with self.lock:
                    self.exp_queue.put([self.s_batch[:],  # ignore the first chuck
                                        self.a_batch[:],  # since we don't have the
                                        self.r_batch[:],  # control over it
                                        done,
                                        {'entropy': self.entropy_record}])
                time.sleep(0.00001)
                with self.lock:
                    print 'exp_queue.empty:'
                    print self.exp_queue.empty()

                # synchronize the network parameters from the coordinator:
                actor_net_params, critic_net_params = self.net_params_queue.get()
                self.actor.set_network_params(actor_net_params)
                self.critic.set_network_params(critic_net_params)

            del self.s_batch[:]
            del self.a_batch[:]
            del self.r_batch[:]
            del self.entropy_record[:]
            print('Agent {} upload the params.'.format(self.agent_id))

