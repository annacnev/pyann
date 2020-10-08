# PyANN

[![Release Version](https://img.shields.io/github/release/annacnev/pyann.svg)](https://github.com/annacnev/pyann/releases/latest) 

A Python implementation of the [RANN](https://github.com/jefferislab/RANN) package for R.

Finds the k nearest neighbours for every point in a given dataset
in O(N log N) time using Arya and Mount's ANN library (v1.1.3). There is
support for approximate as well as exact searches, fixed radius searches
and bd as well as kd trees.

This package implements nearest neighbors for the Euclidean (L2) metric.

For further details on the underlying ANN library, see http://www.cs.umd.edu/~mount/ANN.

For further details on the R implementation, see [RANN](https://github.com/jefferislab/RANN).

## Installation
### Released versions
The recommendation is to install the released version from [PyPI](https://pypi.org) by doing:

```
pip install pyann
```

## Feedback
Please feel free to:

* submit suggestions and bug-reports at: <https://github.com/annacnev/pyann/issues>
* send pull requests after forking: <https://github.com/annacnev/pyann/>
* e-mail the maintainer: <annanev@umich.edu>

## Copyright and License
see [COPYRIGHT](https://github.com/annacnev/pyann/blob/master/COPYRIGHT) and [LICENSE](LICENSE) files for copyright and license information.
