"""This script starts an iteractive window, where a custom digit can be given.
The digit is then analyzed by the neural network."""

# TODO: add correct implementation
# TODO: neural net has to be exported by `main.py` first !

# ------------------------------------------------------------------------------

import PySimpleGUI as sg


# Style ------------------------------------------------------------------------

buttonColors = {
    0: ('black', 'white'),
    1: ('white', 'black'),
}

fontHeader = ('Helvetica', 30)
fontNormal = ('Helvetica', 20)


# Pixels -----------------------------------------------------------------------

def makeButtonKey(i: int, j: int) -> str:
    return f'pixel-{i}-{j}'


def extractButtonKey(key: str) -> tuple[bool, int, int]:
    try:
        key[:5] == 'pixel'
        i = int(key[6])
        j = int(key[8])
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
        pixels,
        range(len(pixels)),
    ))


def updatePixel(i: int, j: int) -> int:
    newValue = 1 if pixels[i][j] == 0 else 0
    pixels[i][j] = newValue
    return newValue


pixels = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
]


# Network ----------------------------------------------------------------------

net = []

netOutput = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
recognizedDigit = 0


# Digits -----------------------------------------------------------------------

def createDigitOutputs():
    layout = []
    for i in range(10):
        layout.append([
            sg.Text(f'{i}:'),
            sg.Text(f'{netOutput[i]}', key=f'digit-{i}'),
        ])
    return layout


def updateDigitOutputs(window: sg.Window):
    for i in range(len(netOutput)):
        window[f'digit-{i}'].update(netOutput[i])


# Layout -----------------------------------------------------------------------

keyDigit = '-DIGIT-'

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
    sg.Text('Es ist eine:'),
    sg.Text(str(recognizedDigit), key=keyDigit),
]]

print('Starting app...')
window = sg.Window(
    'Ziffern erkennen', layout,
    element_justification='center',
    element_padding=(0, 0),
    font=fontNormal,
)


# Event Loop -------------------------------------------------------------------

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    (isPixel, i, j) = extractButtonKey(event)
    if isPixel:
        newValue = updatePixel(i, j)
        window[event].update(
            text=newValue,
            button_color=buttonColors[newValue],
        )

    updateDigitOutputs(window)

    # TODO: load and run calculation from neural network

window.close()
