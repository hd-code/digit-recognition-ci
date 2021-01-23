# Diese Datei enthält die Aktivierungsfunktion für das neuronale Netz

# ------------------------------------------------------------------------------

from math import exp
from typing import Callable

import numpy as np

# ------------------------------------------------------------------------------

# Aktivierungsfunktion für skalare Werte (Sigmoid- oder logistische Funktion)
def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))

# Ableitung der Aktivierungsfunktion für skalare Werte
def sigmoidDerivative(x: float) -> float:
    sig = sigmoid(x)
    return sig * (1 - sig)

# ------------------------------------------------------------------------------

# Aktivierungsfunktion für Vektoren
calc: Callable[[np.ndarray], np.ndarray] = np.vectorize(sigmoid)

# Ableitung der Aktivierungsfunktion für Vektoren
calcDerivative: Callable[[np.ndarray], np.ndarray] = np.vectorize(sigmoidDerivative)

# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    input: np.ndarray = np.array([-10, -5, -1, -0.5, 0, 0.5, 1, 5, 10], 'f')

    want = np.array([0, 0.007, 0.269, 0.378, 0.5, 0.622, 0.731, 0.993, 1])
    wantDeriv = np.array([0, 0.007, 0.197, 0.235, 0.25, 0.235, 0.197, 0.007, 0])

    got = calc(input).round(3)
    gotDeriv = calcDerivative(input).round(3)

    assert np.array_equal(got, want)
    assert np.array_equal(gotDeriv, wantDeriv)

    print('activation.py => tests succeeded')
