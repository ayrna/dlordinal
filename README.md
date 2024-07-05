# Deep learning utilities library

`dlordinal` is an open-source Python toolkit focused on deep learning with ordinal methodologies.

| Overview  |                                                                                                                                          |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD** | [![!codecov](https://img.shields.io/codecov/c/github/ayrna/dlordinal?label=codecov&logo=codecov)](https://codecov.io/gh/ayrna/dlordinal) [![!docs](https://readthedocs.org/projects/dlordinal/badge/?version=latest&style=flat)](https://dlordinal.readthedocs.io/en/latest/)  [![!python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/) |
| **Code**  | [![![pypi]](https://img.shields.io/pypi/v/dlordinal)](https://pypi.org/project/dlordinal/2.0.0/) [![![binder]](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ayrna/dlordinal/main?filepath=tutorials) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Linter: Ruff](https://img.shields.io/badge/Linter-Ruff-brightgreen?style=flat-square)](https://github.com/charliermarsh/ruff)                     |


## Table of Contents
- [⚙️ Installation](#%EF%B8%8F-installation)
- [📖 Documentation](#-documentation)
- [Collaborating](#collaborating)
    - [Guidelines for code contributions](#guidelines-for-code-contributions)

## ⚙️ Installation

`dlordinal v2.1.0` is the last version supported by Python 3.8, Python 3.9 and Python 3.10.

The easiest way to install `dlordinal` is via `pip`:

```bash
pip install dlordinal
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

   losses
   datasets
   distributions
   layers
   metrics
   sklearn_integration
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
