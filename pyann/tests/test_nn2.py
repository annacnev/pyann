from unittest import TestCase
import pyann
import numpy as np


class TestNN2(TestCase):

    def test_basic_use_1(self):
        s = pyann.nn2(np.matrix([[1, 0],
                                 [2, 0]]),

                      np.matrix([[1.01, 0],
                                 [3, 0],
                                 [4.0, 0]]),
                      k=1)

        nn_match = np.array_equal(s.nn_idx,
                                  np.matrix([1, 2, 2]).reshape(3, 1)
                                  )

        dist_match = np.array_equal(s.nn_dists,
                                    np.matrix([0.010000000000000009, 1, 2],
                                              dtype=np.double).reshape(3, 1)
                                    )

        self.assertTrue(
            np.array_equal(np.array([nn_match, dist_match]),
                           np.array([True, True])
                           )
        )

    def test_basic_use_2(self):
        data = np.matrix([[-0.626453810742332, 1.51178116845085, 0.918977371608218],
                          [0.183643324222082, 0.389843236411431, 0.782136300731067],
                          [-0.835628612410047, -0.621240580541804, 0.0745649833651906],
                          [1.59528080213779, -2.2146998871775, -1.98935169586337],
                          [0.329507771815361, 1.12493091814311, 0.61982574789471],
                          [-0.820468384118015, -0.0449336090152309, -0.0561287395290008],
                          [0.487429052428485, -0.0161902630989461, -0.155795506705329],
                          [0.738324705129217, 0.9438362106853, -1.47075238389927],
                          [0.575781351653492, 0.821221195098089, -0.47815005510862],
                          [-0.305388387156356, 0.593901321217509, 0.417941560199702]])

        query = np.matrix([[1.35867955152904, -0.164523596253587, 0.398105880367068],
                           [-0.102787727342996, -0.253361680136508, -0.612026393250771],
                           [0.387671611559369, 0.696963375404737, 0.341119691424425],
                           [-0.0538050405829051, 0.556663198673657, -1.12936309608079],
                           [-1.37705955682861, -0.68875569454952, 1.43302370170104],
                           [-0.41499456329968, -0.70749515696212, 1.98039989850586],
                           [-0.394289953710349, 0.36458196213683, -0.367221476466509],
                           [-0.0593133967111857, 0.768532924515416, -1.04413462631653],
                           [1.10002537198388, -0.112346212150228, 0.569719627442413],
                           [0.763175748457544, 0.881107726454215, -0.135054603880824]])

        nn_idx = np.matrix([[7, 2, 9, 5, 10],
                            [7, 6, 3, 9, 10],
                            [5, 2, 10, 9, 7],
                            [9, 8, 7, 6, 10],
                            [3, 6, 10, 2, 1],
                            [2, 3, 10, 6, 5],
                            [6, 10, 7, 9, 3],
                            [9, 8, 7, 6, 10],
                            [7, 2, 5, 9, 10],
                            [9, 5, 7, 2, 10]]).reshape(10, 5)

        nn_dists = np.matrix(
            [[1.04301819567468, 1.35481071903339, 1.53376883835267, 1.66464626768377, 1.82885863701822],
             [0.782785251535478, 0.931412898425681, 1.06948509416105, 1.27793083718641, 1.3489757618116],
             [0.514019762075842, 0.574844301957827, 0.704879811283639, 0.849722423887674, 0.874908208472254],
             [0.943635726058299, 0.945472961596115, 1.25256907321914, 1.44966319723756, 1.56806650285981],
             [1.46393847125353, 1.71518967014802, 1.95552547157996, 2.00569777023645, 2.38117935086023],
             [1.73157573316917, 1.95360681279363, 2.03639936314308, 2.179643484087, 2.40067304459157],
             [0.66791447696244, 0.822782991518318, 0.983420985126367, 1.07789743302845, 1.16696237017602],
             [0.852326186446708, 0.921390522381496, 1.30532148465522, 1.4890400761415, 1.49288843286122],
             [0.954406845715277, 1.06633538007411, 1.45844529983923, 1.49813576432115, 1.58019302443234],
             [0.395496580572611, 0.904080753178077, 0.938940949941509, 1.19098185772077, 1.23698083599426]],
            dtype=np.double).reshape(10, 5)

        s = pyann.nn2(data, query, k=5)

        nn_match = np.array_equal(s.nn_idx,
                                  nn_idx
                                  )

        dist_match = np.array_equal(s.nn_dists.round(8),
                                    nn_dists.round(8)
                                    )

        self.assertTrue(
            np.array_equal(np.array([nn_match, dist_match]),
                           np.array([True, True])
                           )
        )

    # NB this fails with the version of ANN distributed with knnFinder v1.0
    def test_nn2_with_identical_point(self):
        data = np.matrix([[1, 0],
                          [2, 0]])

        query = np.matrix([[1, 0],
                           [3, 0],
                           [4, 0]])

        s = pyann.nn2(data, query, k=1)

        nn_dists = np.matrix([[0],
                              [1],
                              [2]], dtype=np.double).reshape(3, 1)

        nn_idx = np.matrix([[1],
                            [2],
                            [2]]).reshape(3, 1)

        nn_match = np.array_equal(s.nn_idx,
                                  nn_idx
                                  )

        dist_match = np.array_equal(s.nn_dists.round(8),
                                    nn_dists.round(8)
                                    )

        self.assertTrue(
            np.array_equal(np.array([nn_match, dist_match]),
                           np.array([True, True])
                           )
        )

    # nn2 with different search / tree types
    def test_tree_search_types(self):
        data = np.matrix(np.random.normal(size=3000)).reshape(1000, 3)
        query = np.matrix(np.random.normal(size=3000)).reshape(1000, 3)

        n_standard = pyann.nn2(data, query, k=5, searchtype='standard')
        n_priority = pyann.nn2(data, query, k=5, searchtype='priority')
        n_bd_standard = pyann.nn2(data, query, k=5, searchtype='standard', treetype='bd')
        n_bd_priority = pyann.nn2(data, query, k=5, searchtype='priority', treetype='bd')

        self.assertTrue(
            np.array_equal(n_standard.to_array(),
                           n_priority.to_array())
        )

        self.assertTrue(
            np.array_equal(n_standard.to_array(),
                           n_bd_standard.to_array())
        )

        self.assertTrue(
            np.array_equal(n_standard.to_array(),
                           n_bd_priority.to_array())
        )

    # nn2 fixed radius with large radius
    def test_fixed_large_radius(self):
        data = np.matrix(np.random.normal(size=3000)).reshape(1000, 3)
        query = np.matrix(np.random.normal(size=3000)).reshape(1000, 3)

        n_standard = pyann.nn2(data, query, k=5, searchtype='standard')
        n_rad = pyann.nn2(data, query, k=5, searchtype='radius', radius=20.0)
        n_bd_rad = pyann.nn2(data, query, k=5, searchtype='radius', radius=20.0, treetype='bd')

        self.assertTrue(
            np.array_equal(n_standard.to_array(), n_rad.to_array())
        )

        self.assertTrue(
            np.array_equal(n_standard.to_array(), n_bd_rad.to_array())
        )

    # matrix with 0 columns
    def test_zero_columns(self):
        data = np.matrix([]).reshape(90, 0)
        with self.assertRaises(ValueError):
            pyann.nn2(data)

    # all NA
    def test_all_na(self):
        data = np.matrix(np.random.normal(size=10)).reshape(5, 2)
        query = np.matrix([np.nan] * 10).reshape(5, 2)
        with self.assertRaises(ValueError):
            pyann.nn2(data, query, k=1)

    # mixture of matrix and vector inputs
    def test_mix_mat_vector(self):
        query = np.random.normal(size=10)
        data = np.matrix(query).reshape(10, 1)
        s = pyann.nn2(data, query, k=1)
        s_inv = pyann.nn2(query, data, k=1)
        s_ind = pyann.nn2(query, query, k=1)

        self.assertTrue(
            np.array_equal(s.to_array(),
                           s_inv.to_array())
        )

        self.assertTrue(
            np.array_equal(s.to_array(),
                           s_ind.to_array())
        )

    # vector inputs give outputs of appropriate length
    def test_vector_input_length(self):
        s = pyann.nn2(data=np.random.normal(size=10), query=np.random.normal(size=5), k=1)
        self.assertTrue(
            len(s.nn_idx) == 5
        )

    # inputs with different dimensions
    def test_input_diff_dim(self):
        data = np.matrix(np.random.normal(size=20)).reshape(10, 2)
        query = np.asarray(data[:, 1])
        with self.assertRaises(pyann.DimensionError):
            pyann.nn2(data, query, k=1)

    def test_wrong_str_phrase_treetype(self):
        with self.assertRaises(ValueError,
                               msg="unrecognized option 'random' for treetype; must be one of ['kd', 'bd']"):
            pyann.nn2(data=np.random.normal(size=10), query=np.random.normal(size=5), treetype='random', k=1)

    def test_wrong_input_type_treetype(self):
        with self.assertRaises(TypeError, msg="treetype must be str; detected <class 'int'>"):
            pyann.nn2(data=np.random.normal(size=10), query=np.random.normal(size=5), treetype=1, k=1)

    def test_wrong_str_phrase_searchtype(self):
        with self.assertRaises(ValueError,
                               msg="unrecognized option 'random' for searchtype; must be one of ['standard', 'priority', 'radius']"):
            pyann.nn2(data=np.random.normal(size=10), query=np.random.normal(size=5), searchtype='random', k=1)

