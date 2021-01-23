# ------------------------------------------------------------------------------

import activation as a

import numpy as np

# ------------------------------------------------------------------------------

class Layer:
    def __init__(self, bias: np.ndarray, weights: np.ndarray):
        self.bias = bias
        self.weights = weights

def init(numOfInputs: int, numOfOutputs: int) -> Layer:
    return Layer(
        np.random.rand(numOfOutputs),
        np.random.rand(numOfOutputs, numOfInputs),
    )

def calc(layer: Layer, input: np.ndarray) -> np.ndarray:
    return a.calc(layer.bias + layer.weights @ input)

# Returns layer output and its derivatives
def calcForTrain(layer: Layer, input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    biased = layer.bias + layer.weights @ input
    return (a.calc(biased), a.calcDerivative(biased))

def applyDelta(layer: Layer, delta: Layer, learnRate: float) -> Layer:
    return Layer(
        layer.bias - delta.bias * learnRate,
        layer.weights - delta.weights * learnRate,
    )

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    layer = init(2, 3)
    delta = init(2, 3)

    result = applyDelta(layer, delta, 0)

    print(result.bias)
    print(result.weights)
