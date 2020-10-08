# PyANN

[![Release Version](https://img.shields.io/github/release/annacnev/pyann.svg)](https://github.com/annacnev/pyann/releases/latest)
[![Documentation Status](https://readthedocs.org/projects/pyann/badge/?version=latest)](https://pyann.readthedocs.io/en/latest/?badge=latest)

Finds the k nearest neighbours for every point in a given dataset
in O(N log N) time using Arya and Mount's ANN library (v1.1.3). There is
support for approximate as well as exact searches, fixed radius searches
and bd as well as kd trees.

This package implements nearest neighbors for the Euclidean (L2) metric.

For further details on the underlying ANN library, see http://www.cs.umd.edu/~mount/ANN.

PyANN was written to be the Python equivalent of the R package RANN. For further details on the R implementation, see [RANN](https://github.com/jefferislab/RANN).

## Requirements

### Python Version

PyANN requires Python>=3.6 due to the use of type annotations in the source code, which was implemented in [Python 3.6](https://www.python.org/dev/peps/pep-0526/#non-goals).

### Dependencies
   * [Numpy](https://numpy.org)
   * [Pandas](https://pandas.pydata.org/pandas-docs/stable/#)

## Installation

### PyPI

The recommendation is to install the latest released version from [PyPI](https://pypi.org) by doing:

```
pip install pyann
```

### Source

To install PyANN from source you need [Cython](https://pypi.org/project/Cython/) and [setuptools](https://pypi.org/project/setuptools/) >=18.0 in addition to the normal dependencies above. Cython can be installed from PyPI:

```
pip install cython
```

In the PyANN directory (same one where you found this file after cloning the git repo), execute:

```
python setup.py install
```
## Documentation

Documentation for PyANN is available at: <https://pyann.readthedocs.io/en/latest/>

## Feedback
Please feel free to:

* submit suggestions and bug-reports at: <https://github.com/annacnev/pyann/issues>
* send pull requests after forking: <https://github.com/annacnev/pyann/>
* e-mail the maintainer: <annanev@umich.edu>

## Copyright and License
see [COPYRIGHT](COPYRIGHT.md) and [LICENSE](LICENSE.md) files for copyright and license information.