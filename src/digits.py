import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------------------------


class Digit:
    def __init__(self, digit: int, kind: str, img: np.ndarray):
        self.digit = digit
        self.kind = kind
        self.img = img

    def __repr__(self):
        return '%s-%s:\n%s\n' % (self.digit, self.kind, self.img)

    def getInputVector(self) -> np.ndarray:
        return self.img.flatten()

    def getTargetVector(self) -> np.ndarray:
        result = np.zeros(10)
        result[self.digit] = 1
        return result


ALL_DIGITS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
ALL_KINDS = {'normal', 'klein', 'digital', 'digital-klein', 'evag'}


# ------------------------------------------------------------------------------


def _loadDigitFromFile(digit: int, kind: str) -> Digit:
    filePath = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'digits',
        str(digit) + '-' + kind + '.csv'
    )
    csvFile = pd.read_csv(filePath, header=None, sep=";")
    img = np.array(csvFile)
    return Digit(digit, kind, img)


def _loadAllDigits() -> list[Digit]:
    digits = []
    for digit in ALL_DIGITS:
        for kind in ALL_KINDS:
            digits.append(_loadDigitFromFile(digit, kind))
    return digits


_allDigits = _loadAllDigits()


# ------------------------------------------------------------------------------


def getDigits(digits=ALL_DIGITS, kinds=ALL_KINDS) -> list[Digit]:
    global _allDigits
    return list(filter(
        lambda digit: digit.digit in digits and digit.kind in kinds,
        _allDigits
    ))


def extractInputAndOutput(digitSet: list[Digit]) -> tuple[list[float], list[float]]:
    inputs = list(map(lambda digit: digit.getInputVector(), digitSet))
    targets = list(map(lambda digit: digit.getTargetVector(), digitSet))
    return (inputs, targets)


# ------------------------------------------------------------------------------
# Testing


if __name__ == '__main__':
    numOfDigits = len(ALL_DIGITS) * len(ALL_KINDS)

    print('\ndigits.py')
    print('  check if all digits were loaded')
    numOfDigitsWant = len(ALL_DIGITS) * len(ALL_KINDS)
    numOfDigitsGot = len(_allDigits)
    assert numOfDigitsGot == numOfDigitsWant, 'got %d digits, expected %d' % (
        numOfDigitsGot, numOfDigitsWant
    )
    print('SUCCESS\n')
