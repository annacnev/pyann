cimport numpy as np
import numpy as np
cimport cython

cdef extern from "NN.cc":
    void get_NN_2Set(double *data, double *query, int *D, int *ND, int *NQ, int *K, double *EPS, int *SEARCHTYPE, int *USEBDTREE, double *SQRAD, int *nn_index, double *distances)


def py_get_NN_2Set(np.ndarray[double, ndim=1] data, np.ndarray[double, ndim=1] query, int D, int ND, int NQ, int K, double EPS, int SEARCHTYPE, int USEBDTREE, double SQRAD, np.ndarray[int, ndim=1] nn_index, np.ndarray[double, ndim=1] distances) -> tuple:
    data = np.ascontiguousarray(data)
    query = np.ascontiguousarray(query)
    nn_index = np.ascontiguousarray(nn_index)
    distances = np.ascontiguousarray(distances)
    results = get_NN_2Set(&data[0], &query[0], &D, &ND, &NQ, &K, &EPS, &SEARCHTYPE, &USEBDTREE, &SQRAD, &nn_index[0], &distances[0])
    return nn_index, distances





