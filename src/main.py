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


NUM_OF_HIDDEN = 20

np.random.seed(0)

net = nn.init(35, NUM_OF_HIDDEN, 10)
error = nn.calcBatchError(net, trainingSetInput, trainingSetTarget)

print('1st net error:', error)

for i in range(9999):
    altNet = nn.init(35, NUM_OF_HIDDEN, 10)
    altError = nn.calcBatchError(altNet, trainingSetInput, trainingSetTarget)
    if altError < error:
        net = altNet
        error = altError
        print(i+2, 'is better:', error)

print('best init net error:', error)

for i in range(99):
    net = nn.trainBatch(net, trainingSetInput, trainingSetTarget, 0.1)

error = nn.calcBatchError(net, trainingSetInput, trainingSetTarget)
print('after training error:', error)

error = nn.calcBatchError(net, testSetInput, testSetTarget)
print('test set error:', error)
