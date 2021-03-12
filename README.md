# Digit Recognition – Computational Intelligence

This repo contains the project files for the course 'Computational Intelligence' at the University of Applied Science Erfurt. The goal is to programm a neural network that recognizes arabic numbers from 0 to 9 in small pixel graphics (5x7).

TODO: Add info about Jupyter

## Installation

- have **Python** *version 3.9 or later* installed
- make sure the **Pip** package manger is installed as well (comes together with python)
- install **Pipenv**, which handles all additional dependencies: `pip install pipenv`
- now go to the project directory and run: `pipenv install`

Now all dependencies were installed and you are ready to go.

## Usage

There are two applications: the **training script** and the **demo app**.

The **training script** is at `src/main.py`. It will load the training and test digits and then create a neural network and perform the training. The results will be shown in various plots during the execution of the script. In the end, the results are stored in `data/simulations/<timestamp>`, where `<timestamp>` is the unix timestamp when the script was run. The results include the trained neural network.

The **demo app** is at `src/app.py`. It will start in an interactive window and load the latest generated neural network from the simulations directory (see above). There is a pixel grid on the left. By clicking a pixel it can be toggled (empty/filled). Next to the grid, the results from the calculations on the neural network are shown. Here you can see, which digit the algorithm did recognize.

*Note:* The scripts have to be run from `pipenv`, otherwise the dependencies will not be available:

- training script: `pipenv run python src/main.py`
- demo app: `pipenv run python src/app.py`

## Project Structure

- `data/`
  - `digits/`: the digits to be recognized by the network in CSV format
  - `digits.numbers`: file for Apple Numbers, which was used to create the digits
  - `simulations/`: not used yet...
- `docs/`: the documentation of the project in german
  - `projektabgabe.pdf`: the final report that was graded
- `src/`
  - `app.py`: the demo app
  - `digits.py`: loads and exposes the digits from `data/digits/`
  - `main.py`: the training script
  - `net/`: the implementation of a neural network as a flexible library

## Testing

All files except the demo app and the main script in `src/` contain a testing section at the end of the file. These tests check if the software behaves as it should.

The tests are executed when a file is run as a main script. If you use a bash shell, you can execute all test by running the following command:

```sh
files=(src/digits.py src/net/[^_]*.py)
for f in $files; do PYTHONPATH=src pipenv run python $f; done
```

To run only a single file's tests, execute:

```sh
PYTHONPATH=src pipenv run python <path-to-file>.py
```
