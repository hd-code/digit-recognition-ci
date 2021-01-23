# ------------------------------------------------------------------------------

import error as e
import layer

import numpy as np

# ------------------------------------------------------------------------------

class Net:
    def __init__(self, hiddenLayer: layer.Layer, outputLayer: layer.Layer):
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer

def init(numOfInputs: int, numOfHiddenNeurons: int, numOfOutputs: int) -> Net:
    return Net(
        layer.init(numOfInputs, numOfHiddenNeurons),
        layer.init(numOfHiddenNeurons, numOfOutputs),
    )

# ------------------------------------------------------------------------------

def calc(net: Net, input: np.ndarray) -> np.ndarray:
    hiddenOutput = layer.calc(net.hiddenLayer, input)
    return layer.calc(net.outputLayer, hiddenOutput)

def calcError(net: Net, input: np.ndarray, want: np.ndarray) -> float:
    output = calc(net, input)
    return e.calc(output, want)

def calcBatch(net: Net, inputs: list[np.ndarray]) -> list[np.ndarray]:
    return list(map(lambda input: calc(net, input), inputs))

def calcBatchError(net: Net, inputs: list[np.ndarray], wants: list[np.ndarray]) -> float:
    errors = list(map(lambda input, want: calcError(net, input, want), inputs, wants))
    return np.mean(errors)

# ------------------------------------------------------------------------------

def train(net: Net, input: np.ndarray, want: np.ndarray, learnRate: float) -> Net:
    delta = _calcDelta(net, input, want)
    return Net(
        layer.applyDelta(net.hiddenLayer, delta.hiddenLayer, learnRate),
        layer.applyDelta(net.outputLayer, delta.outputLayer, learnRate),
    )

def trainBatch(net: Net, inputs: list[np.ndarray], wants: list[np.ndarray], learnRate: float) -> Net:
    pass

# ------------------------------------------------------------------------------

def _calcDelta(net: Net, input: np.ndarray, want: np.ndarray) -> Net:
    (hOutput, hActDeriv) = layer.calcForTrain(net.hiddenLayer, input)
    (oOutput, oActDeriv) = layer.calcForTrain(net.outputLayer, hOutput)

    deltaOutputB = e.calcDerivative(oOutput, want) * oActDeriv
    deltaOutputW = _mulVecAndTVec(deltaOutputB, hOutput)

    deltaHiddenB = net.outputLayer.weights.transpose() @ deltaOutputB * hActDeriv
    deltaHiddenW = _mulVecAndTVec(deltaHiddenB, input)

    return Net(
        layer.Layer(deltaHiddenB, deltaHiddenW),
        layer.Layer(deltaOutputB, deltaOutputW),
    )

def _mulVecAndTVec(vector: np.ndarray, transposedVector: np.ndarray) -> np.ndarray:
    return np.array(list(map(lambda x: x * transposedVector, vector)))

# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    np.random.seed(1)

    net = init(2, 4, 2)
    inputs = [
        np.array([0,0]),
        np.array([0,1]),
        np.array([1,0]),
        np.array([1,1]),
    ]
    wants = [
        np.array([1,0]),
        np.array([0,1]),
        np.array([0,1]),
        np.array([0,0]),
    ]

    numOfCases = len(inputs)

    for _ in range(1):
        for i in range(numOfCases):
            input = inputs[i]
            want = wants[i]
            net = train(net, input, want, .1)
        result = calcBatch(net, inputs)
        error = calcBatchError(net, inputs, wants)
        # print(result)
        print(error)
