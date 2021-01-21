import math
import numpy as np

# ------------------------------------------------------------------------------

# vector => vector
def activation(x):
    return np.array(map(sigmoid, x))

# vector => vector
def activationDerivative(x):
    return np.array(map(sigmoidDerivative, x))

# ------------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoidDerivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# - Testing --------------------------------------------------------------------

if __name__ == '__main__':
    input = [-10, -5, -1, -0.5, 0, 0.5, 1, 5, 10]

    want = [0, 0.007, 0.269, 0.378, 0.5, 0.622, 0.731, 0.993, 1]
    wantDeriv = [0, 0.007, 0.197, 0.235, 0.25, 0.235, 0.197, 0.007, 0]

    got = map(lambda x: round(x * 1000) / 1000, activation(input))
    gotDeriv = map(lambda x: round(x * 1000) / 1000, activationDerivative(input))

    assert got == want, 'sigmoid failed'
    assert gotDeriv == wantDeriv, 'sigmoid derivative failed'
