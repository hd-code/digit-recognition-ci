"""This script starts an iteractive window, where a custom digit can be given.
The digit is then analyzed by the neural network."""

# TODO: add correct implementation
# TODO: neural net has to be exported by `main.py` first !

# ------------------------------------------------------------------------------

import PySimpleGUI as sg

# ------------------------------------------------------------------------------


def makeButtonKey(i: int, j: int) -> str:
    return f'pixel-{i}-{j}'


def extractButtonKey(key: str) -> tuple[bool, int, int]:
    try:
        event[:5] == 'pixel'
        i = int(event[6])
        j = int(event[8])
        return (True, i, j)
    except:
        return (False, 0, 0)


NUM_OF_ROWS = 7
NUM_OF_COLS = 5

pixels = []
pixelButtons = []

for i in range(NUM_OF_ROWS):
    pixels.append([])
    pixelButtons.append([])

    for j in range(NUM_OF_COLS):
        key = makeButtonKey(i, j)
        value = 0
        pixels[i].append(0)
        pixelButtons[i].append(
            sg.Button(value, key=key)
        )


# ------------------------------------------------------------------------------


sg.theme('Dark Blue 3')

window = sg.Window('Ziffern erkennen', pixelButtons)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    (isPixel, i, j) = extractButtonKey(event)
    if isPixel:
        newValue = 1 if pixels[i][j] == 0 else 0

        pixels[i][j] = newValue
        window[makeButtonKey(i, j)].update(newValue)

        # TODO: update Button Colors
        # TODO: load and run calculation from neural network

window.close()
