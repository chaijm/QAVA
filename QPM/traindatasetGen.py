import os
import numpy as np
import cv2
import h5py

# generate train dataset

INPUT_W = 96
INPUT_H = 54
INPUT_D = 3
INPUT_SEQ = 24
DEVICE = 2

def load_score(filename, device):
    if device == 1:
        _reader = open('normalizequality/' + filename + '_ph_vmaf.log', 'r')
    elif device == 2:
        _reader = open('normalizequality/' + filename + '_tv_vmaf.log', 'r')

    _array = []
    for _line in _reader:
        _sp = _line.strip('\n').split(',')
        # _sp = _sp[6:16]
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

def saveh5f(filename, x, y, z):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('X', data=x)
    h5f.create_dataset('Y', data=y)
    h5f.create_dataset('Z', data=z)
    h5f.close()
    print 'save done'


def event_loop():
    _dirs = os.listdir('train/')
    _x_array, _y_array, _z_array = [], [], []
    for _dir in _dirs:
        _files = os.listdir('train/' + _dir + '/')

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
            for _file in _p:
                x = np.roll(x, -1, axis=0)
                _img = load_image('train/' + _dir + '/' + _dir + '_' + str(_file) + '.png')
                x[-1] = _img
                _index += 1
                if _index % (INPUT_SEQ / 4 * 2) == 0:
                    _z_index = _index / (INPUT_SEQ / 4 * 2)
                    if len(z) > _z_index:
                        _x_array.append(x)
                        _y_array.append(y)
                        _z_array.append(z[_z_index])

            d = d+1
    return np.array(_x_array), np.array(_y_array), np.array(_z_array)


def main():
    x, y, z = event_loop()
    saveh5f('train_2s.h5', x, y, z)


if __name__ == '__main__':
    main()
