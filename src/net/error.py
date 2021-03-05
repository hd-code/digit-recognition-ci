"""Holds the error function. Shows, how much the neural network is off with its
prediction.

This error function is minimized during training.
"""

# ------------------------------------------------------------------------------

import numpy as np

# ------------------------------------------------------------------------------


def mse(actual: np.ndarray, expected: np.ndarray) -> float:
    """Error function: Mean squared error

    Calculates the error between the actual and the expected vector.
    """
    diff = actual - expected
    diff = diff * diff
    return diff.sum() / len(diff)


def mseDerivative(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Derivative of Error function: Mean squared error

    Calculates the derivative for each element in the vector and returns them
    all together in a vector as well.
    """
    return 2 * (actual - expected) / len(actual)


# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    inputAct = np.array([1, 1,   1,  1])
    inputExp = np.array([1, 0, -0.5, 0.5])

    wantErr = 0.875
    wantDeriv = [0, 0.5, 0.75, 0.25]

    print('\nerror.py:')

    print('  test error function')
    gotErr = mse(inputAct, inputExp)
    assert np.array_equal(gotErr, wantErr)

    print('  test error function derivative')
    gotDeriv = mseDerivative(inputAct, inputExp)
    assert np.array_equal(gotDeriv, wantDeriv)

    print('SUCCESS\n')
