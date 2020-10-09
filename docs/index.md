# PyANN

[![Release Version](https://img.shields.io/github/release/annacnev/pyann.svg)](https://github.com/annacnev/pyann/releases/latest)
[![PyPI version](https://badge.fury.io/py/pyann.svg)](https://badge.fury.io/py/pyann)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyann?color=limegreen&label=pypi%20downloads)
[![Build Status](https://img.shields.io/travis/annacnev/pyann/master.svg?label=Linux%20CI&logo=travis&logoColor=white)](https://travis-ci.org/annacnev/pyann)
![Windows Build Status](https://img.shields.io/appveyor/ci/annacnev/pyann/master.svg?label=Windows%20CI&logo=appveyor&logoColor=white)
[![Documentation Status](https://readthedocs.org/projects/pyann/badge/?version=latest)](https://pyann.readthedocs.io/en/latest/?badge=latest)


Finds the k nearest neighbours for every point in a given dataset
in $O(N \log {N})$ time using Arya and Mount's ANN library (v1.1.3). There is
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

The recommendation is to install the latest released version from [PyPI](https://pypi.org/project/pyann/0.0.1/) by doing:

```
pip install pyann
```

### Source

#### macosx / linux

To install PyANN from source you need [Cython](https://pypi.org/project/Cython/) and [setuptools](https://pypi.org/project/setuptools/) >=18.0 in addition to the normal dependencies above. Cython can be installed from PyPI:

```
pip install cython
```

In the PyANN directory (same one where you found this file after cloning the git repo), first execute:
```
make
```
then:
```
python setup.py install
```
#### windows

Requires 
- [MinGW](http://www.mingw.org/wiki/getting_started)
- [Visual Studio](https://visualstudio.microsoft.com/downloads/)
- [Python Tools for Visual Studio](https://docs.microsoft.com/en-us/visualstudio/python/installing-python-support-in-visual-studio?view=vs-2019)


Install g++, gcc, and mingw32-make

```
mingw-get install gcc g++ mingw32-make
```

Install all required dependencies using `pip`:

```
"%PYTHONPATH%/python.exe" -m pip install --upgrade setuptools cython numpy pandas
```
where `%PYTHONPATH%` is the path to your `python.exe`.

In the PyANN directory (same one where you found this file after cloning the git repo), first execute:

```
%MAKE%
```
where `%MAKE%` is the path to your `mingw32-make.exe`

then:

```
"%PYTHONPATH%/python.exe" -m pip install .
```

## Feedback
Please feel free to:

* submit suggestions and bug-reports at: <https://github.com/annacnev/pyann/issues>
* send pull requests after forking: <https://github.com/annacnev/pyann/>
* e-mail the maintainer: <annanev@umich.edu>

## Copyright and License
see [COPYRIGHT](COPYRIGHT.md) and [LICENSE](LICENSE.md) files for copyright and license information.
