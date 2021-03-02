# Digit Recognition – Computational Intelligence

This repo contains the project files for the course 'Computational Intelligence' at the University of Applied Science Erfurt.
The goal is to programm a neural network that recognizes arabic numbers from 0 to 9 in small pixel graphics (5x7).

## Installation

- have **Python** *version 3.9 or later* installed
- make sure the Pip package manger is installed as well (comes together with python)
- install Pipenv, which handles all additional dependencies: `pip install pipenv`
- now go to the project directory and run: `pipenv install`

Now all dependencies were installed and you are ready to go.

## Usage

The main script is `src/main.py`. By running it, you see the final simulation of this project.

## Project Structure

- `data/`
  - `digits/`: the digits to be recognized by the network in CSV format
  - `digits.numbers`: file for Apple Numbers, which was used to create the digits
  - `simulations/`: not used yet...
- `docs/`:
  - `projekt.*`: the final report that was graded
- `src/`
  - `digits.py`: loads and exposes the digits from `data/digits/`
  - `main.py`: the main simulation
  - `net/`: the implementation of a neural network as a flexible library

## Testing

All files in `src/` contain a testing section at the end of the file. These tests check if the software behaves as it should.

The tests are executed when a file is run as a main script. If you use a bash shell, you can execute all test by running the following command:

```sh
for f in $(find src -name "*.py" -not -path "src/main.py"); do PYTHONPATH=src python $f; done
```
