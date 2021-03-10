from net.main import Net, init, calc
from net.layer import Layer

import json
import numpy as np
import os

# ------------------------------------------------------------------------------


def saveNet(net: Net, filepath: str):
    np.save(filepath + '_' + 'hiddenBias', net.hiddenLayer.bias)
    np.save(filepath + '_' + 'hiddenWeights', net.hiddenLayer.weights)
    np.save(filepath + '_' + 'outputBias', net.outputLayer.bias)
    np.save(filepath + '_' + 'outputWeights', net.outputLayer.weights)


def loadNet(filepath: str) -> Net:
    return Net(
        Layer(
            np.load(filepath + '_' + 'hiddenBias' + '.npy'),
            np.load(filepath + '_' + 'hiddenWeights' + '.npy'),
        ),
        Layer(
            np.load(filepath + '_' + 'outputBias' + '.npy'),
            np.load(filepath + '_' + 'outputWeights' + '.npy'),
        ),
    )


# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    filepath = 'tmp'
    while os.path.exists(filepath):
        filepath = filepath + filepath

    print('\nfile.py')
    print('  store and load net again')
    tmpNet = init(2, 3, 1)
    saveNet(tmpNet, filepath)
    net = loadNet(filepath)

    print('  both nets should be equal')
    assert tmpNet == net

    print('  do calculation on the loaded net')
    calc(net, np.array([0, 0]))

    print('  remove the dumpfiles')
    os.remove(filepath + '_' + 'hiddenBias' + '.npy')
    os.remove(filepath + '_' + 'hiddenWeights' + '.npy')
    os.remove(filepath + '_' + 'outputBias' + '.npy')
    os.remove(filepath + '_' + 'outputWeights' + '.npy')

    print('SUCCESS\n')
