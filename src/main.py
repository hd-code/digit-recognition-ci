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

np.random.seed(0)

net = nn.init(35, 20, 10)

error = nn.calcBatchError(net, trainingSetInput, trainingSetTarget)
print(error)

for i in range(100):
    net = nn.trainBatch(net, trainingSetInput, trainingSetTarget, 0.1)

error = nn.calcBatchError(net, trainingSetInput, trainingSetTarget)
print(error)

error = nn.calcBatchError(net, testSetInput, testSetTarget)
print(error)
