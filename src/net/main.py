"""Defines the data structure for a neural network and methods to do operations
on that network."""

# NOTE: This file is called `main.py` because otherwise it would collide with
#       the module name `net`

import numpy as np

import net.error as e
import net.layer as layer

# ------------------------------------------------------------------------------


class Net:
    """A neural network

    This is the data structure representing a neural network. It has one hidden
    and one output layer.
    """

    def __init__(self, hiddenLayer: layer.Layer, outputLayer: layer.Layer):
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer

    def __repr__(self):
        return '-- hidden layer: \n%s,\n\n-- output layer: \n%s\n' % (self.hiddenLayer, self.outputLayer)

    def __eq__(self, other):
        return str(self) == str(other)


def init(numOfInputs: int, numOfHiddenNeurons: int, numOfOutputs: int) -> Net:
    """Creates a new neural network instance with the given number of neurons on
    the corresponding layer. The weights and biases are initialized randomly."""
    return Net(
        layer.init(numOfInputs, numOfHiddenNeurons),
        layer.init(numOfHiddenNeurons, numOfOutputs),
    )


# ------------------------------------------------------------------------------


def calc(net: Net, input: np.ndarray) -> np.ndarray:
    """Runs a calculation on a neural network.

    - `input` is the input vector
    - this function returns a vector as well
    """
    hiddenOutput = layer.calc(net.hiddenLayer, input)
    return layer.calc(net.outputLayer, hiddenOutput)


def calcError(net: Net, input: np.ndarray, target: np.ndarray) -> float:
    """Runs a calculation on a neural network. It then calculates the error
    between calculated and target output."""
    output = calc(net, input)
    return e.mse(output, target)


def calcBatch(net: Net, inputs: list[np.ndarray]) -> list[np.ndarray]:
    """Runs a calculation on a neural network for a batch of input vectors."""
    return list(map(lambda input: calc(net, input), inputs))


def calcBatchError(net: Net, inputs: list[np.ndarray], targets: list[np.ndarray]) -> float:
    """Runs a calculation on a neural network for a batch of input vectors. It
    then calculates the error between calculated and target outputs."""
    errors = list(map(lambda input, target: calcError(
        net, input, target), inputs, targets))
    return np.mean(errors)


# ------------------------------------------------------------------------------


def train(net: Net, input: np.ndarray, target: np.ndarray, learnRate: float) -> Net:
    """Executes one training iteration on the neural network for a single set of
    input and target values.

    The returned neural network is a new instance. The original network is not
    modified.
    """
    delta = _calcDelta(net, input, target)
    return Net(
        layer.applyDelta(net.hiddenLayer, delta.hiddenLayer, learnRate),
        layer.applyDelta(net.outputLayer, delta.outputLayer, learnRate),
    )


def trainBatch(net: Net, inputs: list[np.ndarray], targets: list[np.ndarray], learnRate: float) -> Net:
    """Executes one training iteration on the neural network for a whole set of
    input and target values.

    This is batch training, so the error for each set is calculated and then the
    average delta for each weight is calculated from that.

    The returned neural network is a new instance. The original network is not
    modified.
    """
    deltas = list(map(lambda input, target: _calcDelta(
        net, input, target), inputs, targets))

    hiddenB = np.array(list(map(lambda delta: delta.hiddenLayer.bias, deltas)))
    hiddenW = np.array(
        list(map(lambda delta: delta.hiddenLayer.weights, deltas)))
    outputB = np.array(list(map(lambda delta: delta.outputLayer.bias, deltas)))
    outputW = np.array(
        list(map(lambda delta: delta.outputLayer.weights, deltas)))

    return Net(
        layer.applyDelta(net.hiddenLayer, layer.Layer(
            hiddenB.mean(0), hiddenW.mean(0)), learnRate),
        layer.applyDelta(net.outputLayer, layer.Layer(
            outputB.mean(0), outputW.mean(0)), learnRate),
    )


# ------------------------------------------------------------------------------


def _calcDelta(net: Net, input: np.ndarray, target: np.ndarray) -> Net:
    """Calculates the delta for both hidden and output layer of the network for
    the given input and target vector."""
    (hOutput, hActDeriv) = layer.calcForTrain(net.hiddenLayer, input)
    (oOutput, oActDeriv) = layer.calcForTrain(net.outputLayer, hOutput)

    deltaOutputB = e.mseDerivative(oOutput, target) * oActDeriv
    deltaOutputW = _mulVecAndVecT(deltaOutputB, hOutput)

    deltaHiddenB = net.outputLayer.weights.transpose() @ deltaOutputB * hActDeriv
    deltaHiddenW = _mulVecAndVecT(deltaHiddenB, input)

    return Net(
        layer.Layer(deltaHiddenB, deltaHiddenW),
        layer.Layer(deltaOutputB, deltaOutputW),
    )


def _mulVecAndVecT(vector: np.ndarray, transposedVector: np.ndarray) -> np.ndarray:
    """Multiplies a vector x with a transposed vector y^T. The result is a
    matrix which has as many rows as x has elements and as many columns as y has
    elements."""
    return np.array(list(map(lambda x: x * transposedVector, vector)))


# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    # seed the random number generator to always get the same values
    np.random.seed(0)

    numOfInputs = 2
    numOfHidden = 4
    numOfOutputs = 3
    learnRate = 0.1

    inputs = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1]),
    ]
    targets = [  # AND, XOR, IMPLICATION
        np.array([0, 0, 1]),
        np.array([0, 1, 1]),
        np.array([0, 1, 0]),
        np.array([1, 0, 1]),
    ]
    testIndex = 0
    iterations = 200  # lower values will fail, because error is not reduced enough

    wantNet = Net(
        layer.Layer(
            np.array([0.09762701, 0.43037873, 0.20552675, 0.08976637]),
            np.array([
                [-0.1526904, 0.29178823],
                [-0.12482558, 0.783546],
                [0.92732552, -0.23311696],
                [0.58345008, 0.05778984],
            ]),
        ),
        layer.Layer(
            np.array([0.13608912, 0.85119328, -0.85792788]),
            np.array([
                [-0.8257414, -0.95956321, 0.66523969, 0.5563135],
                [0.7400243,  0.95723668, 0.59831713, -0.07704128],
                [0.56105835, -0.76345115, 0.27984204, -0.71329343],
            ]),
        ),
    )
    wantOutputs = [
        np.array([0.44495748, 0.89176025, 0.22361304]),
        np.array([0.38477885, 0.9075474, 0.20472936]),
        np.array([0.51340532, 0.89694437, 0.21663046]),
        np.array([0.45141924, 0.91327932, 0.19811374]),
    ]
    wantErrSingle = 0.5320000724762278
    wantErrBatch = 0.3736861371567025

    snapshotTrainedNet = '''-- hidden layer: \nbias: [ 0.23129396  0.34364699  0.20485942 -0.0131533 ]\nweights: \n[[-0.23316034  0.33850732]\n [-0.18397432  0.6077061 ]\n [ 0.92454783 -0.13785743]\n [ 0.60993843  0.01353776]],\n\n-- output layer: \nbias: [-0.17051793  0.26982272 -0.073488  ]\nweights: \n[[-0.97584854 -1.10252765  0.52139186  0.42501474]\n [ 0.41146236  0.57705615  0.23285626 -0.40848657]\n [ 1.04694174 -0.17441979  0.67780059 -0.31265852]]\n'''

    print('\nnet.py')

    print('  test init net')
    net = init(numOfInputs, numOfHidden, numOfOutputs)
    assert str(net) == str(wantNet)

    # --------------------------------------------------------------------------

    print('  test calc single')
    got = calc(net, inputs[testIndex])
    assert np.array_equal(got.round(3), wantOutputs[testIndex].round(3))

    print('  test calc single error')
    got = calcError(net, inputs[testIndex], targets[testIndex])
    assert got == wantErrSingle, 'want: %s, got: %s' % (wantErrSingle, got)

    print('  test calc batch')
    got = calcBatch(net, inputs)
    for i in range(len(got)):
        assert np.array_equal(got[i].round(3), wantOutputs[i].round(3))

    print('  test calc batch error')
    got = calcBatchError(net, inputs, targets)
    assert got == wantErrBatch, 'want: %s, got: %s' % (wantErrBatch, got)

    # --------------------------------------------------------------------------

    print('  test train net with single data value')
    origOutput = calc(net, inputs[testIndex])
    gotNet = net
    gotErr = calcError(gotNet, inputs[testIndex], targets[testIndex])

    print('    - error should decrease with every training')
    for i in range(iterations):
        gotNet = train(gotNet, inputs[testIndex],
                       targets[testIndex], learnRate)
        newErr = calcError(gotNet, inputs[testIndex], targets[testIndex])
        assert newErr < gotErr
        gotErr = newErr

    print('    - net should produce different output after training')
    got = calc(gotNet, inputs[testIndex])
    assert not np.array_equal(got.round(1), origOutput.round(1))

    print('    - net should produce correct output after training')
    assert np.array_equal(got.round(), targets[testIndex])

    # --------------------------------------------------------------------------

    print('  test train net with batch data')
    origOutput = calcBatch(net, inputs)
    gotNet = net
    gotErr = calcBatchError(gotNet, inputs, targets)

    print('    - error should decrease with every training')
    for i in range(iterations):
        gotNet = trainBatch(gotNet, inputs, targets, learnRate)
        newErr = calcBatchError(gotNet, inputs, targets)
        assert newErr < gotErr
        gotErr = newErr

    print('    - net should produce different output after training')
    got = calcBatch(gotNet, inputs)
    for i in range(len(got)):
        assert not np.array_equal(got[i].round(1), origOutput[i].round(1))

    print('    - net should produce correct output after training')
    for i in range(len(got)):
        assert not np.array_equal(got[i].round(1), targets[i])

    print('    - snapshot of trained network matches')
    assert snapshotTrainedNet == str(
        gotNet), 'snapshot did not match, got:\n%s' % gotNet

    # --------------------------------------------------------------------------

    print('SUCCESS\n')
