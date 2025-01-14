# Welcome to dlordinal

`dlordinal` is a Python library that unifies many recent deep ordinal classification methodologies available in the literature. Developed using PyTorch as underlying framework, it implements the top performing state-of-the-art deep learning techniques for ordinal classification problems. Ordinal approaches are designed to leverage the ordering information present in the target variable. Specifically, it includes loss functions, various output layers, dropout techniques, soft labelling methodologies, and other classification strategies, all of which are appropriately designed to incorporate the ordinal information. Furthermore, as the performance metrics to assess novel proposals in ordinal classification depend on the distance between target and predicted classes in the ordinal scale, suitable ordinal evaluation metrics are also included.

The latest `dlordinal` release is `v2.3.0`.

| Overview  |                                                                                                                                          |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD** | [![Run Tests](https://github.com/ayrna/dlordinal/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/ayrna/dlordinal/actions/workflows/run_tests.yml) [![!codecov](https://img.shields.io/codecov/c/github/ayrna/dlordinal?label=codecov&logo=codecov)](https://codecov.io/gh/ayrna/dlordinal) [![!docs](https://readthedocs.org/projects/dlordinal/badge/?version=latest&style=flat)](https://dlordinal.readthedocs.io/en/latest/)  [![!python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/) |
| **Code**  | [![![pypi]](https://img.shields.io/pypi/v/dlordinal)](https://pypi.org/project/dlordinal/2.0.0/) [![![binder]](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ayrna/dlordinal/main?filepath=tutorials) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Linter: Ruff](https://img.shields.io/badge/Linter-Ruff-brightgreen?style=flat-square)](https://github.com/charliermarsh/ruff)                     |


## Table of Contents
- [Welcome to dlordinal](#welcome-to-dlordinal)
  - [Table of Contents](#table-of-contents)
  - [⚙️ Installation](#️-installation)
  - [🚀 Getting started](#-getting-started)
    - [Loading an ordinal benchmark dataset](#loading-an-ordinal-benchmark-dataset)
    - [Training a CNN model using the `skorch` library](#training-a-cnn-model-using-the-skorch-library)
  - [📖 Documentation](#-documentation)
  - [Collaborating](#collaborating)
    - [Guidelines for code contributions](#guidelines-for-code-contributions)

## ⚙️ Installation

`dlordinal v2.3.0` is the last version, supported from Python 3.8 to Python 3.12.

The easiest way to install `dlordinal` is via `pip`:

```bash
pip install dlordinal
```

## 🚀 Getting started

The best place to get started with `dlordinal` is the [tutorials directory](https://github.com/ayrna/dlordinal/tree/main/tutorials).

Below we provide a quick example of how to use some elements of `dlordinal`, such as a dataset, a loss function or some metrics.

### Loading an ordinal benchmark dataset

The FGNet is a well-known benchmark dataset that is commonly used to benchmark ordinal classification methodologies. The dataset is composed of facial images and is labelled with different age categories. It can be downloaded and loaded into Python by simply using the `dlordinal.datasets.FGNet` class.

```python
import numpy as np
from dlordinal.datasets import FGNet
from torchvision.transforms import Compose, ToTensor

fgnet_train = FGNet(
    root="./datasets",
    train=True,
    target_transform=np.array,
    transform=Compose([ToTensor()]),
)
fgnet_test = FGNet(
    root="./datasets",
    train=False,
    target_transform=np.array,
    transform=Compose([ToTensor()]),
)

```

### Training a CNN model using the `skorch` library

This example shows how to train a CNN model using the `NeuralNetClassifier` from the `skorch` library and the `TriangularCrossEntropy` from `dlordinal` as optimisation criterion.

```python
import numpy as np
from dlordinal.datasets import FGNet
from dlordinal.losses import TriangularCrossEntropyLoss
from dlordinal.metrics import amae, mmae
from skorch import NeuralNetClassifier
from torch import nn
from torch.optim import Adam
from torchvision import models
from torchvision.transforms import Compose, ToTensor

# Download the FGNet dataset
fgnet_train = FGNet(
    root="./datasets",
    train=True,
    target_transform=np.array,
    transform=Compose([ToTensor()]),
)
fgnet_test = FGNet(
    root="./datasets",
    train=False,
    target_transform=np.array,
    transform=Compose([ToTensor()]),
)

num_classes_fgnet = len(fgnet_train.classes)

# Model
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes_fgnet)

# Loss function
loss_fn = TriangularCrossEntropyLoss(num_classes=num_classes_fgnet)

# Skorch estimator
estimator = NeuralNetClassifier(
    module=model,
    criterion=loss_fn,
    optimizer=Adam,
    lr=1e-3,
    max_epochs=25,
)

estimator.fit(X=fgnet_train, y=fgnet_train.targets)
train_probs = estimator.predict_proba(fgnet_train)
test_probs = estimator.predict_proba(fgnet_test)

# Metrics
amae_metric = amae(np.array(fgnet_test.targets), test_probs)
mmae_metric = mmae(np.array(fgnet_test.targets), test_probs)
print(f"Test AMAE: {amae_metric}, Test MMAE: {mmae_metric}")
```

## 📖 Documentation

`Sphinx` is a documentation generator tool that is commonly used in the Python ecosystem. It allows developers to write documentation in a markup language called reStructuredText (reST) and generates HTML, PDF, and other formats from it. Sphinx provides a powerful and flexible way to document code, making it easier for developers to create comprehensive and user-friendly documentation for their projects.

To document `dlordinal`, it is necessary to install all documentation dependencies:

```bash
pip install -e '.[docs]'
```

Then access the `docs/` directory:

```bash
docs/
↳ api.rst
↳ conf.py
↳ distributions.rst
↳ references.bib
↳ ...
```

If a new module is created in the software project, the `api.rst` file must be modified to include the name of the new module:

```plaintext
.. _api:

=============
API Reference
=============

This is the API for the **dlordinal** package.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   datasets
   dropout
   output_layers
   losses
   metrics
   wrappers
   soft_labelling
   ***NEW_MODULE***
```

Afterwards, a new file in `.rst` format associated to the new module must be created, specifying the automatic inclusion of documentation from the module files containing a docstring, and the inclusion of the bibliography if it exists within any of them.

```bash
docs/
↳ api.rst
↳ conf.py
↳ distributions.rst
↳ new_module.rst
↳ references.bib
↳ ...
```

```plaintext
.. _new_module:

New Module
==========

.. automodule:: dlordinal.new_module
    :members:

.. footbibliography::

```

Finally, if any new bibliographic citations have been added, they should be included in the `references.bib` file.

## Collaborating

Code contributions to the dlordinal project are welcomed via pull requests.
Please, contact the maintainers (maybe opening an issue) before doing any work to make sure that your contributions align with the project.

### Guidelines for code contributions

* You can clone the repository and then install the library from the local repository folder:

```bash
git clone git@github.com:ayrna/dlordinal.git
pip install ./dlordinal
```

* In order to set up the environment for development, install the project in editable mode and include the optional dev requirements:
```bash
pip install -e '.[dev]'
```
* Install the pre-commit hooks before starting to make any modifications:
```bash
pre-commit install
```
* Write code that is compatible with all supported versions of Python listed in the `pyproject.toml` file.
* Create tests that cover the common cases and the corner cases of the code.
* Preserve backwards-compatibility whenever possible, and make clear if something must change.
* Document any portions of the code that might be less clear to others, especially to new developers.
* Write API documentation as docstrings.

## Citation

If you use dlordinal we would appreciate a citation of the following [paper](https://arxiv.org/abs/2407.17163)

```bibtex
@article{berchez2024dlordinal,
  title={dlordinal: A Python package for deep ordinal classification},
  author={B{\'e}rchez-Moreno, Francisco and Ayll{\'o}n-Gavil{\'a}n, Rafael and Vargas, V{\'\i}ctor M and Guijo-Rubio, David and Herv{\'a}s-Mart{\'\i}nez, C{\'e}sar and Fern{\'a}ndez, Juan C and Guti{\'e}rrez, Pedro A},
  journal={Neurocomputing},
  pages={129305},
  year={2024},
  publisher={Elsevier},
  doi={doi.org/10.1016/j.neucom.2024.129305}
}
```
