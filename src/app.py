"""This script starts an interactive window, where a custom digit can be given.
The digit is then analyzed by the neural network."""

import os

import numpy as np
import PySimpleGUI as sg

import net as nn


# Pixels -----------------------------------------------------------------------
# The input pixel field on the left

PIXELS = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
]


def updatePixel(i: int, j: int) -> int:
    newValue = 1 if PIXELS[i][j] == 0 else 0
    PIXELS[i][j] = newValue
    return newValue


def makeButtonKey(i: int, j: int) -> str:
    return f'-PIXEL-{i}-{j}'


def extractButtonKey(key: str) -> tuple[bool, int, int]:
    """Checks if the given key belongs to a pixel and if so, it extracts the
    position (i and j)."""
    try:
        trimmed = key.strip('-')
        parts = trimmed.split('-')
        assert parts[0] == 'PIXEL'
        i = int(parts[1])
        j = int(parts[2])
        return (True, i, j)
    except:
        return (False, 0, 0)


def createPixelButtons():
    return list(map(
        lambda row, i: list(map(
            lambda value, j: sg.Button(
                value,
                button_color=buttonColors[value],
                key=makeButtonKey(i, j),
                size=(2, 1),
                pad=(0, 0),
            ),
            row,
            range(len(row)),
        )),
        PIXELS,
        range(len(PIXELS)),
    ))


# Network ----------------------------------------------------------------------
# The neural network for the output on the right.

srcPath = os.path.dirname(os.path.realpath(__file__))
netPath = os.path.join(srcPath, '..', 'data', 'cache', 'final-net.pkl')

NET = nn.load(netPath)
KEY_DIGIT = '-DIGIT-'


def calcNetOutput():
    return nn.calc(NET, np.array(PIXELS).flatten())


def getIndexOfHighest(values: list[float]):
    """Returns the index of the highest value in a list."""
    if len(values) <= 0:
        return -1

    result = 0
    highest = values[0]
    for i in range(len(values)):
        value = values[i]
        if highest < value:
            highest = value
            result = i

    return result


def getNetUpdates():
    """Returns a dict with all keys and their updated values."""
    netOutput = calcNetOutput()

    result = {}
    for i in range(len(netOutput)):
        value = netOutput[i]
        result[f'-OUTPUT-{i}-'] = f'{int(value * 100)}'.rjust(3, '_')

    digit = getIndexOfHighest(netOutput)
    result[KEY_DIGIT] = digit

    return result


def createDigitOutputs():
    updateValues = getNetUpdates()
    del updateValues[KEY_DIGIT]

    result = []
    for key, value in updateValues.items():
        i = len(result)
        result.append([
            sg.Text(f'{i}:'),
            sg.Text(value, key=key),
            sg.Text('%'),
        ])

    return result


# Create Window ----------------------------------------------------------------
# Starts the application and initializes the window

buttonColors = {
    0: ('black', 'white'),
    1: ('white', 'black'),
}
fontHeader = ('Helvetica', 30)
fontNormal = ('Helvetica', 20)

layout = [[
    sg.Column(
        [
            [sg.Text('Eingabe', font=fontHeader)],
            [sg.Column(createPixelButtons(), pad=(0, 40))],
        ],
        element_justification='center',
        vertical_alignment='top',
        pad=(40, 0),
    ),
    sg.Column(
        [
            [sg.Text('Ausgabe', font=fontHeader)],
            [sg.Column(createDigitOutputs(), pad=(0, 20))],
        ],
        element_justification='center',
        vertical_alignment='top',
        pad=(40, 0),
    ),
], [
    sg.Text('Es ist eine:', font=fontHeader),
    sg.Text(str(0), key=KEY_DIGIT, font=fontHeader),
]]

print('Starting app...')

window = sg.Window(
    'Ziffern erkennen', layout,
    element_justification='center',
    element_padding=(0, 0),
    font=fontNormal,
    finalize=True,
)

updateValues = getNetUpdates()
for key, value in updateValues.items():
    window[key].update(value)

# Event Loop -------------------------------------------------------------------
# Listens for events (e.g. clicks on the pixels) and updates the window.

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    (isPixel, i, j) = extractButtonKey(event)
    if isPixel:
        newValue = updatePixel(i, j)
        window[event].update(  # update clicked pixel
            text=newValue,
            button_color=buttonColors[newValue],
        )

        updateValues = getNetUpdates()
        for key, value in updateValues.items():
            window[key].update(value)  # update output values

window.close()  # close window once the event loop is stopped
