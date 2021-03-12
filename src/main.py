"""This is the main script, that trains a neural network for digit recognition."""

# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import digits as dg
import net as nn

# ------------------------------------------------------------------------------

DATA = {
    'all': dg.extractInputAndOutput(dg.getDigits()),
    'training': dg.extractInputAndOutput(dg.getDigits(kinds={'normal', 'klein', 'digital', 'digital-klein'})),
    'test': dg.extractInputAndOutput(dg.getDigits(kinds={'evag'})),
}

# ------------------------------------------------------------------------------

numOfHiddens = {5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90}
numOfNets = 100

input = DATA['all']['input']
output = DATA['all']['output']

initNets = {
    'net': [],
    'numOfHidden': [],
    'error': []
}

for numOfHidden in numOfHiddens:
    np.random.seed(0)  # to create reproduceable results

    for _ in range(numOfNets):
        net = nn.init(35, numOfHidden, 10)
        error = nn.calcBatchError(net, input, output)

        initNets['net'].append(net)
        initNets['numOfHidden'].append(numOfHidden)
        initNets['error'].append(error)

INIT_NETS = pd.DataFrame(initNets)

# ------------------------------------------------------------------------------

display(INIT_NETS.groupby('numOfHidden').min('error').sort_values('error'))
INIT_NETS.plot.scatter('numOfHidden', 'error', ylim=(0, 0.7))

# ------------------------------------------------------------------------------

numOfHiddens = {35, 70, 20, 25, 15, 30, 5}
epoches = 1000
learnRate = 0.1

inputTrain = DATA['training']['input']
outputTrain = DATA['training']['output']

inputTest = DATA['test']['input']
outputTest = DATA['test']['output']

trainHistory = {
    'net': [],
    'numOfHidden': [],
    'errorTrain': [],
    'errorTest': [],
}

for numOfHidden in numOfHiddens:
    net = INIT_NETS[INIT_NETS.numOfHidden == numOfHidden].nsmallest(
        1, 'error').iloc[0].net

    for i in range(epoches):
        net = nn.trainBatch(net, inputTrain, outputTrain, learnRate)
        errorTrain = nn.calcBatchError(net, inputTrain, outputTrain)
        errorTest = nn.calcBatchError(net, inputTest, outputTest)

        trainHistory['net'].append(net)
        trainHistory['numOfHidden'].append(numOfHidden)
        trainHistory['errorTrain'].append(errorTrain)
        trainHistory['errorTest'].append(errorTest)

TRAIN_HIDDEN = pd.DataFrame(trainHistory)

# ------------------------------------------------------------------------------

display(TRAIN_HIDDEN.groupby('numOfHidden').min(
    'errorTrain').sort_values('errorTrain'))
for numOfHidden in {35, 30, 15}:
    TRAIN_HIDDEN[TRAIN_HIDDEN.numOfHidden == numOfHidden].plot.line(
        y={'errorTrain', 'errorTest'}, ylim=(0, 0.2), title=numOfHidden)

# ------------------------------------------------------------------------------

startNet = INIT_NETS[INIT_NETS.numOfHidden == 15].iloc[0].net
learnRates = {0.1, 0.01, 0.001}
epoches = 10000

inputTrain = DATA['training']['input']
outputTrain = DATA['training']['output']

inputTest = DATA['test']['input']
outputTest = DATA['test']['output']

trainHistory = {
    'net': [],
    'learnRate': [],
    'errorTrain': [],
    'errorTest': [],
}

for learnRate in learnRates:
    net = startNet

    for i in range(epoches):
        net = nn.trainBatch(net, inputTrain, outputTrain, learnRate)
        errorTrain = nn.calcBatchError(net, inputTrain, outputTrain)
        errorTest = nn.calcBatchError(net, inputTest, outputTest)

        trainHistory['net'].append(net)
        trainHistory['learnRate'].append(learnRate)
        trainHistory['errorTrain'].append(errorTrain)
        trainHistory['errorTest'].append(errorTest)

TRAIN_LR = pd.DataFrame(trainHistory)

# ------------------------------------------------------------------------------

display(TRAIN_LR.groupby('learnRate').min(
    'errorTrain').sort_values('errorTrain'))
TRAIN_LR.groupby('learnRate').plot.line(
    y={'errorTrain', 'errorTest'}, ylim=(0, 0.4))

# ------------------------------------------------------------------------------

learnRate = 0.1
targetError = 0.05

inputTrain = DATA['training']['input']
outputTrain = DATA['training']['output']

inputTest = DATA['test']['input']
outputTest = DATA['test']['output']

trainHistory = TRAIN_LR[TRAIN_LR.learnRate == learnRate].drop('learnRate', 1)
net = trainHistory.iloc[-1].net
error = nn.calcBatchError(net, inputTrain, outputTrain)

while error > targetError:
    net = nn.trainBatch(net, inputTrain, outputTrain, learnRate)
    errorTrain = nn.calcBatchError(net, inputTrain, outputTrain)
    errorTest = nn.calcBatchError(net, inputTest, outputTest)

    trainHistory = trainHistory.append({
        'net': net,
        'errorTrain': errorTrain,
        'errorTest': errorTest,
    }, ignore_index=True)

    error = errorTrain

TRAIN_FINAL = trainHistory

# ------------------------------------------------------------------------------

TRAIN_FINAL.describe()
