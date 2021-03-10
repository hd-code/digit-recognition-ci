"""This is the main script, that trains a neural network for digit recognition."""

# ------------------------------------------------------------------------------

import digits as dg
import net as nn

import numpy as np

# ------------------------------------------------------------------------------


trainingSet = dg.getDigits(
    kinds={'normal', 'klein', 'digital', 'digital-klein'}
)
testSet = dg.getDigits(kinds={'evag'})

(trainingSetInput, trainingSetTarget) = dg.extractInputAndOutput(trainingSet)
(testSetInput, testSetTarget) = dg.extractInputAndOutput(testSet)


# ------------------------------------------------------------------------------


NUM_OF_HIDDENS = (5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100)
NUM_OF_INITIAL_NETS = 100
NUM_OF_TRAINS = 10

LEARN_RATE = 0.1


def getBestGeneratedNet(numOfHidden: int, numOfNets: int) -> nn.Net:
    np.random.seed(0)

    net = nn.init(7*5, numOfHidden, 10)
    err = nn.calcBatchError(net, trainingSetInput, trainingSetTarget)
    for i in range(numOfNets - 1):
        altNet = nn.init(7*5, numOfHidden, 10)
        altErr = nn.calcBatchError(altNet, trainingSetInput, trainingSetTarget)
        if altErr < err:
            net = altNet
            err = altErr
    return net


initialNets = []
for numOfHidden in NUM_OF_HIDDENS:
    net = getBestGeneratedNet(numOfHidden, NUM_OF_INITIAL_NETS)
    initialNets.append([net])

for i in range(len(initialNets)):
    for j in range(NUM_OF_TRAINS):
        origNet = initialNets[i][j]
        nextNet = nn.trainBatch(origNet, trainingSetInput,
                                trainingSetTarget, LEARN_RATE)
        initialNets[i].append(nextNet)


bestNetIndex = -1
bestNetError = 9999
for i in range(len(initialNets)):
    net = initialNets[i][0]
    error = nn.calcBatchError(net, trainingSetInput, trainingSetTarget)
    if error < bestNetError:
        bestNetError = error
        bestNetIndex = i
print('number of hidden neurons:',
      NUM_OF_HIDDENS[bestNetIndex], 'error:', bestNetError)


bestNetIndex = -1
bestNetError = 9999
for i in range(len(initialNets)):
    net = initialNets[i][len(initialNets[i]) - 1]
    error = nn.calcBatchError(net, trainingSetInput, trainingSetTarget)
    if error < bestNetError:
        bestNetError = error
        bestNetIndex = i
print('number of hidden neurons:',
      NUM_OF_HIDDENS[bestNetIndex], 'error:', bestNetError)

# net = nn.init(35, NUM_OF_HIDDEN, 10)
# error = nn.calcBatchError(net, trainingSetInput, trainingSetTarget)

# print('1st net error:', error)

# for i in range(9999):
#     altNet = nn.init(35, NUM_OF_HIDDEN, 10)
#     altError = nn.calcBatchError(altNet, trainingSetInput, trainingSetTarget)
#     if altError < error:
#         net = altNet
#         error = altError
#         print(i+2, 'is better:', error)

# print('best init net error:', error)

# for i in range(99):
#     net = nn.trainBatch(net, trainingSetInput, trainingSetTarget, 0.1)

# error = nn.calcBatchError(net, trainingSetInput, trainingSetTarget)
# print('after training error:', error)

# error = nn.calcBatchError(net, testSetInput, testSetTarget)
# print('test set error:', error)
