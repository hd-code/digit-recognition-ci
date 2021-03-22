"""This script loads the digits from the data/digits/ directory. It offers
methods to filter these digits and transform them to different formats."""

import json
import os

import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------


ALL_DIGITS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
ALL_KINDS = {'normal', 'normal-klein', 'digital', 'digital-klein', 'evag'}


# ------------------------------------------------------------------------------


class Digit:
    """Helper class that represents a single digit."""

    def __init__(self, digit: int, kind: str, img: np.ndarray):
        self.digit = digit
        self.kind = kind
        self.img = img

    def __repr__(self):
        return '%s-%s:\n%s\n' % (self.digit, self.kind, self.img)

    def getInputVector(self) -> np.ndarray:
        return self.img.flatten()

    def getOutputVector(self) -> np.ndarray:
        result = np.zeros(len(ALL_DIGITS))
        result[self.digit] = 1
        return result


# ------------------------------------------------------------------------------


digitsDir = os.path.join(os.path.dirname(__file__), '..', 'data', 'digits')


def _loadDigitFromFile(digit: int, kind: str) -> Digit:
    """Loads a single digit from a file on disk."""
    fileName = f'{str(digit)}-{kind}.csv'
    filePath = os.path.join(digitsDir, fileName)
    csvFile = pd.read_csv(filePath, header=None, sep=";")
    img = np.array(csvFile)
    return Digit(digit, kind, img)


def _loadAllDigits() -> list[Digit]:
    """Loads all digits in the digits directory."""
    digits = []
    for digit in ALL_DIGITS:
        for kind in ALL_KINDS:
            digits.append(_loadDigitFromFile(digit, kind))
    return digits


_allDigits = _loadAllDigits()  # all digits are cached in this variable


# ------------------------------------------------------------------------------


def getDigits(digits=ALL_DIGITS, kinds=ALL_KINDS) -> list[Digit]:
    """Returns a list of digits, that meet the given filters.

    By default, all digits are returned. However, a set of `digits` or a set of
    `kinds` can be specified. By doing so, only the digits that belong to the
    sets are returned.
    """
    global _allDigits
    return list(filter(
        lambda digit: digit.digit in digits and digit.kind in kinds,
        _allDigits
    ))


def extractInputAndOutput(digitSet: list[Digit]) -> dict[str, list[np.ndarray]]:
    """Returns a dict where 'input' is a list of all input vectors for the given
    digit set and 'output' is a list of the corresponding output vectors."""
    inputs = list(map(lambda digit: digit.getInputVector(), digitSet))
    outputs = list(map(lambda digit: digit.getOutputVector(), digitSet))
    return {
        'input': inputs,
        'output': outputs,
    }


# ------------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    numOfDigits = len(ALL_DIGITS) * len(ALL_KINDS)

    print('\ndigits.py')

    print('  check if all digits were loaded')
    wantNumOfDigits = len(ALL_DIGITS) * len(ALL_KINDS)
    gotNumOfDigits = len(_allDigits)
    assert gotNumOfDigits == wantNumOfDigits, 'got %d digits, expected %d' % (
        gotNumOfDigits, wantNumOfDigits
    )

    print('  check filter for digits')
    wantNumOfDigits = len(ALL_KINDS)

    gotDigits = getDigits(digits={3})
    gotNumOfDigits = len(gotDigits)

    assert gotNumOfDigits == wantNumOfDigits, 'got %d digits, expected %d'

    print('  check filter for digit set (kind)')
    wantNumOfDigits = len(ALL_DIGITS)

    gotDigits = getDigits(kinds={'evag'})
    gotNumOfDigits = len(gotDigits)

    assert gotNumOfDigits == wantNumOfDigits, 'got %d digits, expected %d'

    print('  check snapshot for extracted input and output')
    wantNumOfDigits = len(ALL_KINDS)

    gotDigits = getDigits(digits={5})
    got = extractInputAndOutput(gotDigits)

    assert len(got['input']) == wantNumOfDigits
    assert len(got['output']) == wantNumOfDigits

    print('SUCCESS\n')
