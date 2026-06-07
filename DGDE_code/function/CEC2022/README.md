## Introduction

* This script runs multiple evolutionary/population-based optimization algorithms, such as DE, SHADE, GA, ES, and PSO, on selected CEC2022 benchmark functions. Each experiment is repeated 30 times in parallel, and the results are saved as joblib pickle files.
* Example result directory: `./results_CEC2014_repeats`

## Directory and Files

* Script path: `./run_30_EA.py`
* Related dependencies in the same directory:

  * `others_algorithms.py`: calls different algorithms and saves the results
  * `all_function.py`: provides benchmark functions and their gradients
  * Local algorithm implementations: `DE.py`, `SHADE.py`, `GA.py`, `ES.py`, `PSOS.py`, etc.

## Environment Requirements

The code has been tested under the following environment:

* Python: 3.9
* Main dependencies:

  * `numpy==1.26.0`
  * `scipy==1.13.1`
  * `pandas==2.3.3`
  * `matplotlib==3.9.2`
  * `joblib==1.5.2`
  * `mealpy==3.0.3`
  * `opfunu==1.0.4`
  * `autograd==1.8.0`
  * `numba==0.60.0`
  * `openpyxl==3.1.5`
  * `requests==2.32.5`

## Configurable Options

The following options can be configured at the beginning of `run_30_EA.py`:

* `budget`: number of iterations/evaluations (epoch)
* `population_size`: population size
* DE/JADE parameters: `wf`, `cr`, `strategy_DE`, `jade_pt`, `jade_ap`
* PGD/gradient-related parameters: `mutate_thresh`, `mutate_strength`, `gradient_thresh`
* Parameters for EGD, ESGD, NGDE, PSO, etc. are also defined at the beginning of the script. Please check the script for the exact variable names.
* `Dimension`: list of dimensions to be tested, e.g., `[1000]`
* `functions`: list of benchmark functions to run, e.g., `["Michalewicz", "dixon_price", "Schwefel", "Vincent"]`
* `verbose`: whether to print running information
