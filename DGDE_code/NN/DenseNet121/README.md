## Introduction

`run_DGDE.py` implements and runs the DGDE algorithm based on DenseNet121. It trains and evaluates the model on subsets of specified datasets. The results are saved in joblib format, and a summary CSV file is also exported.

## Directory and Files

* Script: `run_DGDE.py`
* Dependency scripts: `ACC_LOSS.py`, `datasets.py`, `models.py`, `DGDE.py`
* Data download directory relative to the current path: `../data`
* Output files:

  * Single-run results in joblib format: `./Results_DGDE_SGD/{Dataset}_{subset_size}/results_*.pkl`
  * Summary CSV file: `./Results_DGDE/subset_parameter_results.csv`

## Environment and Dependencies

Python 3.8+ is recommended.

Main Python packages:

* `torch`
* `torchvision`
* `numpy`
* `pandas`
* `scikit-learn`
* `joblib`

## Data

`datasets.py` creates or downloads the datasets to `../data`, relative to the parent directory of the `DenseNet121` folder.

The following datasets are supported by default:

* `MNIST`
* `FashionMNIST`
* `CIFAR10`
* `CIFAR100`
* `STL10`
* `SVHN`

To use another data path, modify the `root='../data'` argument in `datasets.py`, or place the datasets under this path.

## Common Configurations

The following variables can be modified directly at the beginning of `run_DGDE.py` to adjust the experiments:

* `batch_size` default: 128
* `num_epoch` default: 100
* `population_size` default: 5
* `lambda_reg`, `niche_size`, `T_per`, and other algorithm hyperparameters
* `F`, `K_base`, `K_0`, `pho_high`, `pho_low`, `FP`, which are DGDE-specific parameters
* `LR`: list of learning rates
* `model`: model name, e.g., `"densenet121"`
* `Data`: list of datasets
* `Sub_Size`: list of subset sizes
* `K_0s`: list of candidate `K_0` values

To run a single configuration, replace the lists with a single value, or manually modify the script or add command-line arguments.
