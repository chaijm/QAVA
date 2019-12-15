import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
import tflearn
import h5py
from tflearn.data_utils import shuffle
import cv2

# generate predicted quality scores using the pre-trained neural network

# in 96x54 pictures
# out chunk's future score
INPUT_W = 96
INPUT_H = 54
INPUT_D = 3
INPUT_SEQ = 24
#output bitrate size
OUTPUT_DIM = 10
DEVICE = 2

KERNEL = 64
DENSE_SIZE = 256
LR_RATE = 1e-4

# need to change to the appropriate model path
PREDICT_MODEL = './best/64_256_0.0001/nn_model_ep_best.ckpt'


class Predictor():
    def __init__(self):
        self.input_w = INPUT_W
        self.input_h = INPUT_H
        self.input_d = INPUT_D
        self.input_seq = INPUT_SEQ
        self.a_dim = OUTPUT_DIM
        self.lr_rate = LR_RATE

        self.kernel = KERNEL
        self.dense_size = DENSE_SIZE
        self.device = -1 #ph:0 or tv:1
        self.video = None
        self.chunkno = 0
        self.video_vmaf = []
        self.chunk_quality = []

        self.x = tf.placeholder(shape=[1, self.input_seq, self.input_h, self.input_w, self.input_d], dtype=tf.float32)
        self.y = tf.placeholder(shape=[1, 2], dtype=tf.float32)
        self.z = tf.placeholder(shape=[1, self.a_dim], dtype=tf.float32)
        self.out = self._create_predictor(self.x, self.y)
        self.acc = tf.sqrt(tflearn.objectives.mean_square(self.out, self.z))
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, PREDICT_MODEL)
        self.x_buff = np.zeros([self.input_seq, self.input_h, self.input_w, self.input_d])

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
            merge_net  = tflearn.merge([net_flat, device], 'concat')

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

    def _get_image(self):
        index = (self.chunkno - 3) * (self.input_seq / 2)
        for p in range(1, self.input_seq + 1):
            self.x_buff = np.roll(self.x_buff, -1, axis=1)
            filename = '../img/' + self.video + '.mp4/' + self.video + '.mp4_' + str(index + p) + '.png'
            img = cv2.imread(filename)

            self.x_buff[-1, :, :, :] = img

        return self.x_buff

    def predict(self, x, y):
        return self.sess.run(self.out, feed_dict={
            self.x: x,
            self.y: y
        })

def load_score(filename, device):
    if device == 1:
        _reader = open('normalizequality/' + filename + '_ph_vmaf.log', 'r')
    elif device == 2:
        _reader = open('normalizequality/' + filename + '_tv_vmaf.log', 'r')

    _array = []
    for _line in _reader:
        _sp = _line.strip('\n').split(',')
        #_sp = _sp[6:16]
        _tmp = []
        for t in _sp:
            if len(t) >= 1:
                _tmp.append(float(t))
        _array.append(np.array(_tmp))
    _array = np.array(_array)
    return _array

def load_image(filename):
    img = cv2.imread(filename)
    return img


pred = Predictor()
#need to change to the names of test video
testdataset = []
# key: *.mp4, value: video name
videodir = {}

_dirs = os.listdir('test/')
for _dir in _dirs:
    _files = os.listdir('test/' + _dir + '/')

    d = 1
    while d <= DEVICE:
        if d == 1:
            y = np.array([1, 0])
        else:
            y = np.array([0, 1])
        z = load_score(_dir, d)
        _p = [int(l.split('_')[-1].split('.')[0]) for l in _files]
        _p.sort()
        x = np.zeros([INPUT_SEQ, INPUT_H, INPUT_W, INPUT_D])
        _index = 0
        dname = videodir[_dir]
        if d == 1:
            _pre = open(dname + '_ph_preValue.txt', 'w')
        else:
            _pre = open(dname + '_tv_preValue.txt', 'w')
        for _file in _p:
            x = np.roll(x, -1, axis=0)
            _img = load_image('test/' + _dir + '/' + _dir + '_' + str(_file) + '.png')
            x[-1] = _img
            _index += 1
            if _index % (INPUT_SEQ / 4 * 2) == 0:
                _z_index = _index / (INPUT_SEQ / 4 * 2)
                if len(z) > _z_index:
                    x = np.reshape(x, [1, INPUT_SEQ, INPUT_H, INPUT_W, INPUT_D])
                    y = np.reshape(y, [1, 2])
                    predict_score = pred.predict(x, y)
                    predict_score = predict_score[0]

                    value = ''
                    value = ','.join(str(i) for i in predict_score)
                    _pre.write(value + '\n')

        _pre.close()
        if d == 1:
            print 'save ' + dname + '_ph_preValue'
        else:
            print 'save ' + dname + '_tv_preValue'

        d = d+1

