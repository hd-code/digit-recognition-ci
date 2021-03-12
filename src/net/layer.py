"""Defines a data structure and methods for a single layers of a neural network.
On this level both output and hidden layer work exactly the same."""

import numpy as np

import net.activation as a

# ------------------------------------------------------------------------------


class Layer:
    """A single layer (hidden or output) in a neural network.

    Properties
    - `bias` is a vector
    - `weights` is a matrix
    """

    def __init__(self, bias: np.ndarray, weights: np.ndarray):
        self.bias = bias
        self.weights = weights

    def __repr__(self):
        return 'bias: %s\nweights: \n%s' % (self.bias, self.weights)

    def __eq__(self, other):
        return str(self) == str(other)


def init(numOfInputs: int, numOfOutputs: int) -> Layer:
    """Creates a new Layer with random bias and weights."""
    return Layer(
        np.random.rand(numOfOutputs) * 2 - 1,
        np.random.rand(numOfOutputs, numOfInputs) * 2 - 1,
    )


# ------------------------------------------------------------------------------


def calc(layer: Layer, input: np.ndarray) -> np.ndarray:
    """Runs a calculation for a single layer."""
    return a.sigmoid(layer.bias + layer.weights @ input)


def calcForTrain(layer: Layer, input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Only used for training. Returns layer output and its derivatives as a tuple."""
    biased = layer.bias + layer.weights @ input
    return (a.sigmoid(biased), a.sigmoidDerivative(biased))


def applyDelta(layer: Layer, delta: Layer, learnRate: float) -> Layer:
    """Applies the delta to the given layer.

    The delta is multiplied by the learning rate and then subtracted from the
    original layer. The function returns a new Layer instance, so the passed
    original layer is not altered.
    """
    return Layer(
        layer.bias - delta.bias * learnRate,
        layer.weights - delta.weights * learnRate,
    )


# ------------------------------------------------------------------------------
# Testing

if __name__ == "__main__":
    # seed the random number generator to always get the same values
    np.random.seed(0)

    numOfInputs = 2
    numOfOutputs = 3
    learnRate = 0.1
    input = np.array([1, 0])

    wantLayer = Layer(
        np.array([0.09762701, 0.43037873, 0.20552675]),
        np.array([[0.08976637, -0.1526904],
                  [0.29178823, -0.12482558],
                  [0.783546, 0.92732552]]),
    )
    wantDelta = Layer(
        np.array([-0.23311696, 0.58345008, 0.05778984]),
        np.array([[0.13608912, 0.85119328],
                  [-0.85792788, -0.8257414],
                  [-0.95956321, 0.66523969]]),
    )
    wantCalcResult = np.array([0.54671173, 0.67308402, 0.72890473])
    wantCalcDeriv = np.array([0.24781801, 0.22004192, 0.19760262])

    snapshotResult = '''bias: [0.1209387  0.37203373 0.19974777]\nweights: \n[[ 0.07615745 -0.23780973]\n [ 0.37758101 -0.04225144]\n [ 0.87950232  0.86080155]]'''

    print('\nlayer.py:')

    print('  test init layer')
    layer = init(2, 3)
    assert layer == wantLayer
    delta = init(2, 3)
    assert delta == wantDelta

    print('  test calc layer')
    got = calc(layer, input)
    assert np.array_equal(got.round(3), wantCalcResult.round(3))

    print('  test calc layer + derivative')
    got = calcForTrain(layer, input)
    assert np.array_equal(got[0].round(3), wantCalcResult.round(3))
    assert np.array_equal(got[1].round(3), wantCalcDeriv.round(3))

    print('  test apply delta (snapshot test)')
    got = applyDelta(layer, delta, learnRate)
    assert str(got) == snapshotResult, 'snapshot did not match, got:\n%s' % got

    print('SUCCESS\n')
