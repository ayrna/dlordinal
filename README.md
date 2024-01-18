# Deep learning utilities library

`dlordinal` is an open-source Python toolkit focused on deep learning with ordinal methodologies. It is compatible with
[scikit-learn](https://scikit-learn.org).

The library includes various modules such as loss functions, models, layers, metrics, and an estimator.

| Overview  |                                                                                                                                          |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD** | [![!codecov](https://img.shields.io/codecov/c/github/ayrna/dlordinal?label=codecov&logo=codecov)](https://codecov.io/gh/ayrna/dlordinal) [![!docs](https://readthedocs.org/projects/dlordinal/badge/?version=latest&style=flat)](https://dlordinal.readthedocs.io/en/latest/)  [![!python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/) |
| **Code**  | [![![pypi]](https://img.shields.io/pypi/v/dlordinal)](https://pypi.org/project/dlordinal/1.0.1/) [![![binder]](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ayrna/dlordinal/main?filepath=tutorials) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Linter: Ruff](https://img.shields.io/badge/Linter-Ruff-brightgreen?style=flat-square)](https://github.com/charliermarsh/ruff)                     |

## ⚙️ Installation

`dlordinal v1.0.1` is the last version supported by Python 3.8, Python 3.9 and Python 3.10.

The easiest way to install `dlordinal` is via `pip`:

    pip install dlordinal


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
