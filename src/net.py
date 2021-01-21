import activation as a
import error as e

import numpy as np

# ------------------------------------------------------------------------------

HIDDEN_LAYER = 'hidden'
OUTPUT_LAYER = 'output'

BIAS = 'bias'
WEIGHTS = 'weights'

# ------------------------------------------------------------------------------

def init(numOfInputs, numOfHiddenNeurons, numOfOutputs):
    return {
        HIDDEN_LAYER: initLayer(numOfInputs, numOfHiddenNeurons),
        OUTPUT_LAYER: initLayer(numOfHiddenNeurons, numOfOutputs),
    }

def initLayer(numOfInputs, numOfOutputs):
    return {
        BIAS: np.random.rand(numOfOutputs),
        WEIGHTS: np.random.rand(numOfOutputs, numOfInputs),
    }

# ------------------------------------------------------------------------------

def calc(net, input):
    oHidden = calcLayer(net[HIDDEN_LAYER], input)
    return calcLayer(net[OUTPUT_LAYER], oHidden)

def calcError(net, input, expected):
    output = calc(net, input)
    return e.error(output, expected)

def calcBatch(net, inputs):
    return map(lambda x: calc(net, x), inputs)

def calcBatchError(net, inputs, expecteds):
    errors = map(lambda x, y: calcError(net, x, y), inputs, expecteds)
    return np.array(errors).mean()

def calcLayer(layer, input):
    biased = layer[BIAS] + layer[WEIGHTS].dot(input)
    return a.activation(biased)

# ------------------------------------------------------------------------------

def train(net, input, expected, learningRate):
    deltaNet = calcDeltaNet(net, input, expected)
    return {
        HIDDEN_LAYER: {
            BIAS: net[HIDDEN_LAYER][BIAS] - learningRate * deltaNet[HIDDEN_LAYER][BIAS],
            WEIGHTS: net[HIDDEN_LAYER][WEIGHTS] - learningRate * deltaNet[HIDDEN_LAYER][WEIGHTS],
        },
        OUTPUT_LAYER: {
            BIAS: net[OUTPUT_LAYER][BIAS] - learningRate * deltaNet[OUTPUT_LAYER][BIAS],
            WEIGHTS: net[OUTPUT_LAYER][WEIGHTS] - learningRate * deltaNet[OUTPUT_LAYER][WEIGHTS],
        }
    }

def calcDeltaNet(net, input, expected):
    # forward pass
    hBiased = net[HIDDEN_LAYER][BIAS] + net[HIDDEN_LAYER][WEIGHTS].dot(input)
    hOutput = a.activation(hBiased)
    hActDeriv = a.activationDerivative(hBiased)

    oBiased = net[OUTPUT_LAYER][BIAS] + net[OUTPUT_LAYER][WEIGHTS].dot(hOutput)
    oOutput = a.activation(oBiased)
    oActDeriv = a.activationDerivative(oBiased)

    errDeriv = e.errorDerivative(oOutput, expected)

    # backward pass
    deltaBOut = errDeriv * oActDeriv
    deltaWOut = mulVecAndTVec(deltaBOut, hOutput)

    deltaBHid = hActDeriv * net[OUTPUT_LAYER][WEIGHTS].transpose().dot(deltaBOut)
    deltaWHid = mulVecAndTVec(deltaBHid, input)

    return {
        HIDDEN_LAYER: {
            BIAS: deltaBHid,
            WEIGHTS: deltaWHid,
        },
        OUTPUT_LAYER: {
            BIAS: deltaBOut,
            WEIGHTS: deltaWOut,
        }
    }

def mulVecAndTVec(vector, transposedVector):
    return np.array(map(lambda x: x * transposedVector, vector))

# - Testing --------------------------------------------------------------------

if __name__ == '__main__':
    net = init(2, 4, 2)
    inputs = [
        np.array([0,0]),
        np.array([0,1]),
        np.array([1,0]),
        np.array([1,1]),
    ]
    expecteds = [
        np.array([1,0]),
        np.array([0,1]),
        np.array([0,1]),
        np.array([0,0]),
    ]

    numOfCases = len(inputs)

    for _ in range(10):
        for i in range(numOfCases):
            input = inputs[i]
            expected = expecteds[i]
            net = train(net, input, expected, 0.1)
            # print calcError(net, input, expected)
        
        print calcBatchError(net, inputs, expecteds)
