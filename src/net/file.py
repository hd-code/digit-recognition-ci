import json
import os

import numpy as np
import pandas as pd

from net.main import Net, init, calc
from net.layer import Layer

# ------------------------------------------------------------------------------


def save(net: Net, filepath: str):
    obj = {'net': [net]}
    df = pd.DataFrame(obj)
    df.to_pickle(filepath)


def load(filepath: str) -> Net:
    df = pd.read_pickle(filepath)
    return df.iloc[0].net


# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    filepathBase = 'tmp'
    filepath = '_' + filepathBase
    while os.path.exists(filepath):
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

    except:
        os.remove(filepath)
