This project implements and reproduces the Dynamic Guided Differential Evolution (DGDE) population-based optimization method, combined with convolutional neural networks, with LeNet5 used as the default model, for training and comparison experiments on MNIST and other datasets. The code includes data loading, model definition, DGDE algorithm implementation, result saving, and visualization tools.

## Main Features

* Implements the DGDE algorithm for neural network training.
* Supported datasets: `MNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100`, `STL10`, `SVHN`.
* Experimental results are saved in `joblib` format, with scripts provided for result summarization and plotting.

## Directory Structure

Key files include:

* `run_DGDE_MNIST.py`: main experiment script.
* `DGDE.py`: implementation of the DGDE algorithm and training loop; saves result dictionaries in `.pkl` format.
* `models.py`: model definitions, including `LeNet5` and a general model creation function.
* `datasets.py`: dataset loading and subset splitting functions; by default, datasets are downloaded to `../data/`.
* `ACC_LOSS.py`: utility functions for evaluation and loss computation.
* `sum_DGDE.py` / `sum_results.py`: result summarization and plotting tools.
* `draw.py`: auxiliary visualization script for comparing multiple algorithms.

## Dependencies

Recommended environment:

* Python 3.8+
* PyTorch / torchvision
* scikit-learn
* pandas
* matplotlib
* joblib
* openpyxl

## Output Description

During each run, the results are saved as `.pkl` files using `joblib.dump`. The default save path is, for example:

```text
./Results_DGDE_SGD/{Dataset}_{subset_size}/results_lr{learning_rate}_K0{K_0}_f_p{fitness_p}_DGDE.pkl
```

The summary CSV file is:

```text
./Results_DGDE/subset_parameter_results.csv
```

This file is written at the end of `run_DGDE_MNIST.py`.

Common fields in the `.pkl` files include:

* `best_test_accs` / `best_test_losses`: global best test accuracy/loss for each epoch.
* `best_valid_accs` / `best_valid_losses`: best validation metrics for each epoch.
* `sparsity_epochs`: sparsity changes over epochs recorded under different thresholds, stored as a dictionary.
* `weight_epochs`: changes in the L1 norm of weights over epochs.
* `num_niches`: number of niches at each epoch.
