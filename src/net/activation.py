"""Implements the activation function for the neural network."""

# ------------------------------------------------------------------------------

from math import exp
from typing import Callable
import numpy as np

# ------------------------------------------------------------------------------


def sigmoidScalar(x: float) -> float:
    """Activation function for scalar values: sigmoid or logistic function"""
    return 1 / (1 + exp(-x))


def sigmoidScalarDerivative(x: float) -> float:
    """Derivative for sigmoid function for scalar values"""
    sig = sigmoidScalar(x)
    return sig * (1 - sig)


sigmoid: Callable[[np.ndarray], np.ndarray] = np.vectorize(sigmoidScalar)
"""Sigmoid function for vectors"""


sigmoidDerivative: Callable[[np.ndarray],
                            np.ndarray] = np.vectorize(sigmoidScalarDerivative)
"""Sigmoid derivative function for vectors"""


# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    input: np.ndarray = np.array([-10, -5, -1, -0.5, 0, 0.5, 1, 5, 10], 'f')

    want = np.array([0, 0.007, 0.269, 0.378, 0.5, 0.622, 0.731, 0.993, 1])
    wantDeriv = np.array(
        [0, 0.007, 0.197, 0.235, 0.25, 0.235, 0.197, 0.007, 0])

    print('\nactivation.py:')

    print('  test sigmoid function (vector version)')
    got = sigmoid(input).round(3)
    assert np.array_equal(got, want)

    print('  test sigmoid function derivative (vector version)')
    gotDeriv = sigmoidDerivative(input).round(3)
    assert np.array_equal(gotDeriv, wantDeriv)

    print('SUCCESS\n')
