# Digit Recognition – Computational Intelligence

This repo contains the project files for the course 'Computational Intelligence' at the University of Applied Science Erfurt.
The goal is to programm a neural network that recognizes arabic numbers from 0 to 9 in small pixel graphics.

## Installation

Following software is needed:

- **Python** version 3.8 or later – with python comes `pip`, which can be used to install all the other dependencies
- **Jupyter Notebooks** for an interactive environment – install with: `pip install jupyter`
- **Numpy** for efficient vector and matrix calculus – install with: `pip install numpy`

## Usage

Just start the Jupyter Notebook, go to the `notebooks/` folder and have fun with the notebooks within. To start the notebook just run this in the terminal: `jupyter run`. Now Jupyter starts and the Notebook will open in your web browser.

## Project Structure

- `data/` contains all kinds of data needed for the project
  - `digits/` contains the digits to be recognized in CSV format
  - `simulations/` not used yet...
- `docs/` contains the project documentation. Most important is `projekt.md`, which is the final document that will be graded.
- `notebooks` contains all Jupyter Notebooks, which execute the actual digit recognition task
- `src/` contains the implementation of the neural network in form of a flexible, reusable software library

## Testing

All files in `src/` contain a testing section at the end of the file. These tests check if the software behaves as it should.

The tests are executed when a file is run as a main script. If you use a bash shell, you can execute all test by running the following command:

```sh
for f in src/*.py; do python $f; done
```
