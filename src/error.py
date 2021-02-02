# Holds the error function, which is minimized.

# ------------------------------------------------------------------------------

import numpy as np

# ------------------------------------------------------------------------------

# Error function: Mean squared error.
# Calculates the error between the actual vector and the expected vector.
def mse(actual: np.ndarray, expected: np.ndarray) -> float:
    diff = actual - expected
    diff = diff * diff
    return diff.sum() / len(diff)

# Derivative of Mean squared error function.
# Calculates the derivative for each element in the vector and returns them all
# in a vector as well.
def mseDerivative(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return 2 * (actual - expected) / len(actual)

# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    inputAct = np.array([1,1,   1,  1])
    inputExp = np.array([1,0,-0.5,0.5])

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
