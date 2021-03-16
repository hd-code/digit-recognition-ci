# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
#  # Training a Neural Network to Recognize Digits
#
#  ## Import Dependencies

# %%
import os

import numpy as np
import pandas as pd

import digits as dg
import net as nn

# %% [markdown]
#  ## Loading Digits from Files

# %%
DATA_PATH = os.path.join('..', 'data', 'cache')

allDigits = dg.getDigits()
trainDigits = dg.getDigits(
    kinds={'normal', 'normal-klein', 'digital', 'digital-klein'})
testDigits = dg.getDigits(kinds={'evag'})

DIGITS = {
    'all': dg.extractInputAndOutput(allDigits),
    'training': dg.extractInputAndOutput(trainDigits),
    'test': dg.extractInputAndOutput(testDigits),
}

# %% [markdown]
#  ## Initialize Networks with Different Number of Hidden Neurons

# %%
# This calculation might take a while.
# See below, how to load the results from cache instead

numOfHiddens = {5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90}
numOfNets = 100

input = DIGITS['all']['input']
output = DIGITS['all']['output']

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


# %%
# store initial networks to cache
INIT_NETS.to_pickle(os.path.join(DATA_PATH, 'init-nets.pkl'))


# %%
# load initial networks from cache
INIT_NETS = pd.read_pickle(os.path.join(DATA_PATH, 'init-nets.pkl'))


# %%
INIT_NETS.plot.scatter('numOfHidden', 'error', ylim=(0, 0.7))


# %%
INIT_NETS.groupby('numOfHidden').min('error').sort_values('error')

# %% [markdown]
#  ## Training Promising Networks

# %%
# This calculation might take a while.
# See below, how to load the results from cache instead

numOfHiddens = {35, 70, 20, 25, 15, 30}
epoches = 1000
learnRate = 0.1

inputTrain = DIGITS['training']['input']
outputTrain = DIGITS['training']['output']

inputTest = DIGITS['test']['input']
outputTest = DIGITS['test']['output']

trainHistory = {
    'net': [],
    'numOfHidden': [],
    'errorTrain': [],
    'errorTest': [],
}

for numOfHidden in numOfHiddens:
    net = INIT_NETS[INIT_NETS.numOfHidden == numOfHidden].sort_values(
        'error').iloc[0].net

    for i in range(epoches):
        net = nn.trainBatch(net, inputTrain, outputTrain, learnRate)
        errorTrain = nn.calcBatchError(net, inputTrain, outputTrain)
        errorTest = nn.calcBatchError(net, inputTest, outputTest)

        trainHistory['net'].append(net)
        trainHistory['numOfHidden'].append(numOfHidden)
        trainHistory['errorTrain'].append(errorTrain)
        trainHistory['errorTest'].append(errorTest)

TRAIN_HIDDEN = pd.DataFrame(trainHistory)


# %%
# store trained promising networks to cache
TRAIN_HIDDEN.to_pickle(os.path.join(DATA_PATH, 'train-hidden.pkl'))


# %%
# load trained promising networks from cache
TRAIN_HIDDEN = pd.read_pickle(os.path.join(DATA_PATH, 'train-hidden.pkl'))


# %%
end = TRAIN_HIDDEN.groupby('numOfHidden').min(
    'errorTrain').sort_values('numOfHidden')
start = INIT_NETS.groupby('numOfHidden').min(
    'error').sort_values('numOfHidden')

diff = end.join(start)
diff.rename(columns={'error': 'errorStart'}, inplace=True)
diff['diff'] = diff.errorStart - diff.errorTrain

diff.sort_values('diff', ascending=False)


# %%
for numOfHidden in {15, 20, 25, 30}:
    TRAIN_HIDDEN[TRAIN_HIDDEN.numOfHidden == numOfHidden].plot.line(
        title=f'Hidden Neurons: {numOfHidden}',
        y={'errorTrain', 'errorTest'}, ylim=(0, 0.2),
        use_index=False,
    )

# %% [markdown]
#  ## Finding the Optimal Learning Rate

# %%
# This calculation might take a while.
# See below, how to load the results from cache instead
numOfHidden = 15
learnRates = {1, 0.1, 0.01, 0.001}
epoches = 10000

startNet = INIT_NETS[INIT_NETS.numOfHidden == numOfHidden].sort_values(
    'error').iloc[0].net

inputTrain = DIGITS['training']['input']
outputTrain = DIGITS['training']['output']

inputTest = DIGITS['test']['input']
outputTest = DIGITS['test']['output']

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


# %%
# store training with different learning rates to cache
TRAIN_LR.to_pickle(os.path.join(DATA_PATH, 'train-learn-rate.pkl'))


# %%
# load training with different learning rates from cache
TRAIN_LR = pd.read_pickle(os.path.join(DATA_PATH, 'train-learn-rate.pkl'))


# %%
for learnRate, df in TRAIN_LR.groupby('learnRate'):
    df.plot.line(
        title=f'Learning Rate: {learnRate}',
        y={'errorTrain', 'errorTest'}, ylim=(0, 0.2),
        use_index=False,
    )


# %%
TRAIN_LR.groupby('learnRate').min('errorTrain').sort_values('errorTrain')

# %% [markdown]
#  ## Final Training of the Network

# %%
# This calculation might take a while.
# See below, how to load the results from cache instead

learnRate = 1
targetError = 0.001

inputTrain = DIGITS['training']['input']
outputTrain = DIGITS['training']['output']

inputTest = DIGITS['test']['input']
outputTest = DIGITS['test']['output']

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


# %%
# store final training round to cache
TRAIN_FINAL.to_pickle(os.path.join(DATA_PATH, 'train-final.pkl'))


# %%
# load final training round from cache
TRAIN_FINAL = pd.read_pickle(os.path.join(DATA_PATH, 'train-final.pkl'))


# %%
TRAIN_FINAL.plot.line(ylim=(0, 0.2))


# %%
TRAIN_FINAL.describe()

# %% [markdown]
#  ## Analyzing the Final Neural Network

# %%
FINAL_NET = TRAIN_FINAL.iloc[-1].net


# %%
# store final network to cache
nn.save(FINAL_NET, os.path.join(DATA_PATH, 'final-net.pkl'))


# %%
# load final network from cache
FINAL_NET = nn.load(os.path.join(DATA_PATH, 'final-net.pkl'))


# %%
net = FINAL_NET
result = {
    'kind': [],
    'error': [],
}

for kind in dg.ALL_KINDS:
    digits = dg.getDigits(kinds={kind})
    inOutputs = dg.extractInputAndOutput(digits)
    error = nn.calcBatchError(net, inOutputs['input'], inOutputs['output'])

    result['kind'].append(kind)
    result['error'].append(error)

pd.DataFrame(result).plot.bar(x='kind', title='Average Error per Set')


# %%
net = FINAL_NET
result = {
    'digit': [],
    'error': [],
}

for digit in dg.ALL_DIGITS:
    digits = dg.getDigits(digits={digit})
    inOutputs = dg.extractInputAndOutput(digits)
    error = nn.calcBatchError(net, inOutputs['input'], inOutputs['output'])

    result['digit'].append(digit)
    result['error'].append(error)

pd.DataFrame(result).plot.bar(x='digit', title='Average Error per Digit')


# %%
net = FINAL_NET
result = {
    'digit': [],
    'error': [],
}

for digit in dg.ALL_DIGITS:
    digits = dg.getDigits(digits={digit}, kinds={'evag'})
    inOutputs = dg.extractInputAndOutput(digits)
    error = nn.calcBatchError(net, inOutputs['input'], inOutputs['output'])

    result['digit'].append(digit)
    result['error'].append(error)

pd.DataFrame(result).plot.bar(
    x='digit', title='Average Error on evag Set per Digit')


# %%
net = FINAL_NET
result = {
    'digit': [],
    'error': [],
}

for digit in dg.ALL_DIGITS:
    digits = dg.getDigits(digits={digit}, kinds={
                          'normal', 'normal-klein', 'digital', 'digital-klein'})
    inOutputs = dg.extractInputAndOutput(digits)
    error = nn.calcBatchError(net, inOutputs['input'], inOutputs['output'])

    result['digit'].append(digit)
    result['error'].append(error)

pd.DataFrame(result).plot.bar(
    x='digit', title='Average Error on Training Set per Digit')


# %%
