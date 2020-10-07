import annlib
import numpy as np
import pandas as pd
import inspect


class DimensionError(Exception):
    pass


class NN2Results:
    def __init__(self, nn_idx, nn_dists):
        self.nn_idx = nn_idx
        self.nn_dists = nn_dists

    def create_rep(self, label, value):
        return pd.DataFrame(value, columns=pd.MultiIndex.from_product([[label], list(range(ncol(value)))])).to_string()

    def to_array(self):
        return np.array([self.nn_idx, self.nn_dists])

    def __repr__(self):
        return '\n'.join([self.create_rep(*x) for x in [('nn_dists', self.nn_dists), ('nn_idx', self.nn_idx)]])


def nrow(mat):
    if isinstance(mat, np.matrix):
        return mat.shape[0]
    else:
        return None


def ncol(mat):
    if isinstance(mat, np.matrix):
        return mat.shape[1]
    else:
        return None


def check_args(value):
    grouped_args = {
        'numpy': ['data', 'query'],
        'numeric': ['k', 'radius', 'eps'],
        'str': ['treetype', 'searchtype']
    }

    mat_arrays = grouped_args['numpy']
    for argv in mat_arrays:
        v = value.locals[argv]
        if isinstance(v, np.matrix):
            pass
        elif isinstance(v, np.ndarray):
            if len(v.shape) == 1:
                pass
            elif v.shape[0] == 1 or v.shape[1] == 1:
                pass
            else:
                raise TypeError(argv + ' must np.matrix or 1D np.ndarray; detected 2D np.ndarray')
        else:
            raise TypeError(argv + ' must be np.matrix or np.ndarray; detected ' + str(type(v)))

    str_args = grouped_args['str']
    str_arg_opts = {'treetype': ["kd", "bd"],
                    'searchtype': ["standard", "priority", "radius"]}
    for argv in str_args:
        v = value.locals[argv]
        opts = str_arg_opts[argv]
        if isinstance(v, str):
            pass
        else:
            raise TypeError(argv + ' must be str; detected ' + str(type(v)))
        if v not in opts:
            raise ValueError("unrecognized option " + "'" + str(v) + "' for " + argv + "; must be one of " + str(opts))

    numeric_args = grouped_args['numeric']
    for argv in numeric_args:
        v = value.locals[argv]
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, np.int64):
            pass
        else:
            raise TypeError(argv + ' must be of type float or int; recieved: ' + str(type(v)))

    return None


def nn2(data, query=None, k=None, treetype="kd", searchtype="standard", radius=0.0, eps=0.0):
    if k is None:
        k = np.minimum(10, len(data))

    if query is None:
        query = data

    check_args(inspect.getargvalues(inspect.currentframe()))

    dimension = ncol(data)
    if dimension is None:
        dimension = 1

    query_dimension = ncol(query)
    if query_dimension is None:
        query_dimension = 1

    ND = nrow(data)
    if ND is None:
        ND = len(data)

    NQ = nrow(query)
    if NQ is None:
        NQ = len(query)

    if dimension != query_dimension:
        raise DimensionError('data dimension does not equal query dimension')

    if k > ND:
        raise ValueError("Cannot find more nearest neighbours than there are points: k < nrow(data)")

    searchtype_opts = ["standard", "priority", "radius"]
    searchtypeInt = searchtype_opts.index(searchtype) + 1

    treetype_opts = ["kd", "bd"]
    treetype = treetype_opts.index(treetype)

    data = np.array(data.transpose(), dtype=np.double).reshape(-1, )
    if len(data) == 0:
        raise ValueError('no data points in data')

    query = np.array(query.transpose(), dtype=np.double).reshape(-1, )

    if len(query) == 0:
        raise ValueError('no query points in data')

    if len([x for x in query if np.isnan(x) or np.isinf(x)]) != 0:
        raise ValueError('NA/NaN/Inf in foreign function call (arg 2)')

    if len([x for x in data if np.isnan(x) or np.isinf(x)]) != 0:
        raise ValueError('NA/NaN/Inf in foreign function call (arg 1)')

    results = annlib.py_get_NN_2Set(data,
                                    query,
                                    int(dimension),
                                    int(ND),
                                    int(NQ),
                                    int(k),
                                    int(eps),
                                    int(searchtypeInt),
                                    int(treetype),
                                    float(radius * radius),
                                    np.zeros(k * NQ, dtype=np.intc),
                                    np.zeros(k * NQ, dtype=np.double))

    results = [np.matrix(x).reshape(int(len(x) / k), int(k)) for x in results]
    nnresults = NN2Results(*results)

    return nnresults
