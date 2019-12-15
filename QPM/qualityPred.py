import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
import tflearn
import h5py
from tflearn.data_utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import sys
# train the predictive quality neural network
# in 96x54 pictures
# out chunk's future score
INPUT_W = 96
INPUT_H = 54
INPUT_D = 3
INPUT_SEQ = 24
#output bitrate size
OUTPUT_DIM = 10

KERNEL = int(sys.argv[1])
DENSE_SIZE = int(sys.argv[2])

EPOCH = 1500
BATCH_SIZE = 64
LR_RATE = float(sys.argv[3])
EARLYSTOP = 50

class Predictor():
    def __init__(self):
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.input_w = INPUT_W
        self.input_h = INPUT_H
        self.input_d = INPUT_D
        self.input_seq = INPUT_SEQ
        self.a_dim = OUTPUT_DIM
        self.lr_rate = LR_RATE

        self.kernel = KERNEL
        self.dense_size = DENSE_SIZE

        self.x = tf.placeholder(shape=[None, self.input_seq, self.input_h, self.input_w, self.input_d], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)
        self.out = self._create_predictor(self.x, self.y)
        self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(tflearn.objectives.mean_square(self.out, self.z))
        self.acc = tf.sqrt(tflearn.objectives.mean_square(self.out, self.z))
        tf.summary.scalar('test loss', self.acc)
        self.sess.run(tf.global_variables_initializer())



    def _cnn_network(self, x, reuse=False):
        with tf.variable_scope('cnn', reuse=reuse):
            network = tflearn.conv_2d(x, KERNEL, 3, activation='relu', regularizer="L2", weight_decay=0.0001)
            network = tflearn.max_pool_2d(network, 3)
            network = tflearn.conv_2d(network, KERNEL, 3, activation='relu', regularizer="L2", weight_decay=0.0001)
            network = tflearn.max_pool_2d(network, 2)
            network = tflearn.fully_connected(network, DENSE_SIZE, activation='relu')

            split_flat = tflearn.flatten(network)
        return split_flat

    def _gru_network(self, network, device):
        with tf.variable_scope('rnn'):
            net = tflearn.gru(network, self.dense_size, return_seq=True)
            net = tflearn.gru(net, self.dense_size, dropout=0.8)
            net_flat = tflearn.flatten(net)
            merge_net = tflearn.merge([net_flat, device], 'concat')
            dense_net = tflearn.fully_connected(merge_net, 2*self.dense_size+1, activation='tanh')
            dense_net = tflearn.fully_connected(dense_net, self.dense_size, activation='tanh')
            pred_value = tflearn.fully_connected(dense_net, self.a_dim, activation='sigmoid')
        return pred_value

    def _create_predictor(self, x, device):
        with tf.variable_scope('predictor'):
            inputs = tflearn.input_data(placeholder=x)
            _split_array = []

            for i in range(INPUT_SEQ):
                tmp_network = tf.reshape(inputs[:, i:i + 1, :, :, :], [-1, self.input_h, self.input_w, self.input_d])
                if i == 0:
                    _split_array.append(self._cnn_network(tmp_network))
                else:
                    _split_array.append(self._cnn_network(tmp_network, True))

            merge_net = tflearn.merge(_split_array, 'concat')
            merge_net = tflearn.flatten(merge_net)
            _count = merge_net.get_shape().as_list()[1]

            net = tf.reshape(merge_net, [-1, _count / self.dense_size, self.dense_size])
            out = self._gru_network(net, device)
        return out

    def train(self, x, y, z):
        self.sess.run(self.optimize, feed_dict={
            self.x: x,
            self.y: y,
            self.z: z
        })

    def accError(self, x, y, z):
        return self.sess.run(self.acc, feed_dict={
            self.x: x,
            self.y: y,
            self.z: z
        })

    def predict(self, x, y):
        return self.sess.run(self.out, feed_dict={
            self.x: x,
            self.y: y
        })

    def merge_result(self, merge_op, x, y, z, step):
        rs = self.sess.run(merge_op, feed_dict={
            self.x: x,
            self.y: y,
            self.z: z
        })
        writer.add_summary(rs, step)



def load_dataset(filename):
    h5f = h5py.File(filename, 'r')
    X = h5f['X']
    Y = h5f['Y']
    Z = h5f['Z']
    X, Y, Z = shuffle(X, Y, Z)
    return X, Y, Z

def save_plot(z_pred, z, j):
    plt.switch_backend('agg')
    plt.figure()
    fig, ax = plt.subplots(z.shape[1], 1, sharex=True, figsize=(10, 16), dpi=100)
    x = np.linspace(0, z.shape[0] - 1, z.shape[0])

    for i in range(z.shape[1]):
        ax[i].grid(True)
        ax[i].plot(x, z[:, i])
        ax[i].plot(x, z_pred[:, i])
    savefig('save/' + str(KERNEL) + '_' + str(DENSE_SIZE) + '_' + str(LR_RATE) + '/' + str(j) + '.png')


if __name__ == "__main__":
    if os.path.exists('best/' + str(KERNEL) + '_' + str(DENSE_SIZE) + '_' + str(LR_RATE) + '.txt'):
        print 'this params has been previously operated.'
        sys.exit()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not os.path.exists('best'):
        os.system('mkdir best')
    if not os.path.exists('log'):
        os.system('mkdir log')
    if not os.path.exists('save'):
        os.system('mkdir save')
    if not os.path.exists('best/' + str(KERNEL) + '_' + str(DENSE_SIZE) + '_' + str(LR_RATE)):
        os.system('mkdir best/' + str(KERNEL) + '_' + str(DENSE_SIZE) + '_' + str(LR_RATE))
    if not os.path.exists('save/' + str(KERNEL) + '_' + str(DENSE_SIZE) + '_' + str(LR_RATE)):
        os.system('mkdir save/' + str(KERNEL) + '_' + str(DENSE_SIZE) + '_' + str(LR_RATE))

    X, Y, Z = load_dataset('./train_2s.h5')
    testX, testY, testZ = load_dataset('./test_2s.h5')
    gpu_options = tf.GPUOptions(allow_growth=True)

    pred = Predictor()
    writer = tf.summary.FileWriter('./netlog/', pred.sess.graph)
    merge_op = tf.summary.merge_all()

    train_len = X.shape[0]
    best_saver = tf.train.Saver()
    _writer = open('log/' + str(KERNEL) + '_' + str(DENSE_SIZE) + '_' + str(LR_RATE) + '.txt', 'w')
    _min_mape, _min_step = 100.0, 0
    for j in range(1, EPOCH + 1):
        i = 0
        while i < train_len - BATCH_SIZE:
            batch_xs, batch_ys, batch_zs = X[i:i+BATCH_SIZE], Y[i:i+BATCH_SIZE], Z[i:i+BATCH_SIZE]
            pred.train(batch_xs, batch_ys, batch_zs)
            i += BATCH_SIZE

        _test_mape = pred.accError(testX, testY, testZ)
        pred.merge_result(merge_op, testX, testY, testZ, j)
        print 'epoch', j, 'rmse', _test_mape

        if _min_mape > _test_mape:
            _min_mape = _test_mape
            _min_step = j

            best_saver.save(pred.sess, 'best/' + str(KERNEL) + '_' + str(DENSE_SIZE) + '_' + str(LR_RATE) + '/nn_model_ep_best.ckpt')

            testZ_pred = pred.predict(testX, testY)
            save_plot(testZ_pred, testZ, j)

            _pre = open('best/' + str(KERNEL) + '_' + str(DENSE_SIZE) + '_' + str(LR_RATE) + 'preValue.txt', 'w')
            for l in range(len(testZ)):
                _pre.write(str(testZ_pred[l]) + '\n')
                _pre.write(str(testZ[l]) + '\n')
                _pre.write('\n')
            _pre.close()

            _best = open('best/' + str(KERNEL) + '_' + str(DENSE_SIZE) + '_' + str(LR_RATE) + '.txt', 'w')
            _best.write(str(_test_mape))
            _best.close()
            print 'new record'
        else:
            if j - _min_step > EARLYSTOP:
                print 'early stop'
                break

        _writer.write(str(j) + ',' + str(_test_mape) + '\n')
        _writer.flush()
    _writer.close()


