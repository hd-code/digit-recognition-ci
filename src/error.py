# Diese Datei enthält die Fehlerfunktion, die für das neuronale Netz minimiert
# werden soll.

# ------------------------------------------------------------------------------

import numpy as np

# ------------------------------------------------------------------------------

# Fehlerfunktion: Mean squared error
def calc(actual: np.ndarray, expected: np.ndarray) -> float:
    diff = actual - expected
    diff = diff * diff
    return diff.sum() / len(diff)

# Ableitung der Fehlerfunktion – berechnet die Ableitung für jeden Index im
# Vektor und gibt die jeweiligen Ableitungen als Vektor im ganzen zurück
def calcDerivative(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return 2 * (actual - expected) / len(actual)

# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    inputAct = np.array([1,1,   1,  1])
    inputExp = np.array([1,0,-0.5,0.5])

    wantErr = 0.875
    wantDeriv = [0, 0.5, 0.75, 0.25]

    gotErr = calc(inputAct, inputExp)
    gotDeriv = calcDerivative(inputAct, inputExp)

    assert np.array_equal(gotErr, wantErr)
    assert np.array_equal(wantDeriv, gotDeriv)

    print('error.py => tests succeeded')
