"""Defines a method to save and load a neural network from a file on disk."""

import json
import os

import numpy as np
import pandas as pd

from net.main import Net, init, calc
from net.layer import Layer

# ------------------------------------------------------------------------------


def save(net: Net, filepath: str):
    """Saves a neural network to a file on disk.

    `filepath` is the path plus the full name of the file including a file
    extension. The recommended file extension is '.pkl'.
    """
    obj = {'net': [net]}
    df = pd.DataFrame(obj)
    df.to_pickle(filepath)


def load(filepath: str) -> Net:
    """Loads a neural network from a file on disk.

    `filepath` is the path plus the full name of the file including the file
    extension.
    """
    df = pd.read_pickle(filepath)
    return df.iloc[0].net


# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    filepathBase = 'tmp'
    filepath = '_' + filepathBase
    while os.path.exists(filepath):  # find a file name that is not used yet
        filepath = filepath + filepathBase

    try:
        print('\nfile.py')
        print('  store and load net again')
        tmpNet = init(2, 3, 1)
        save(tmpNet, filepath)
        net = load(filepath)

        print('  both nets should be equal')
        assert tmpNet == net

        print('  do calculation on the loaded net')
        calc(net, np.array([0, 0]))

        print('  remove the dumpfile')
        os.remove(filepath)

        print('SUCCESS\n')

    except:  # remove the dumpfile in any case
        os.remove(filepath)
