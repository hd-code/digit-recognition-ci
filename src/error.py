# vector => vector => number
def error(actual, expected):
    diffs = (actual - expected) ** 2
    return sum(diffs) / len(diffs)

# vector => vector => vector
def errorDerivative(actual, expected):
    return 2 * (actual - expected) / len(actual)

# - Testing --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np

    inputAct = np.array([1,1,   1,  1])
    inputExp = np.array([1,0,-0.5,0.5])

    wantErr = 0.875
    wantErrDeriv = [0, 0.5, 0.75, 0.25]

    gotErr = error(inputAct, inputExp)
    gotDeriv = errorDerivative(inputAct, inputExp)

    assert gotErr == wantErr
    for i in range(len(wantErrDeriv)):
        assert gotDeriv[i] == wantErrDeriv[i], 'error deriv failed at index: ' + str(i)
