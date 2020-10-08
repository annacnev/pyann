<span style = "font-weight:700;font-size:25px;color:#cc397b">pyann.nn2</span>*<span style="font-size:16px">(data, query=None, k=None, treetype='kd', searchtype='standard', radius=0.0, eps=0.0)</span>*

# 
## pyann.nn2

### Overview
#### Nearest Neighbour Search
Uses a kd-tree to find the p number of near neighbours for each point in an input/output dataset. The advantage of the kd-tree is that it runs in O(M log M) time.

The Pyann package utilizes the Approximate Near Neighbor (ANN) C++ library, which can give the exact near neighbours or approximate near neighbours to within a specified error bound.  

For more information on the ANN library please visit http://www.cs.umd.edu/~mount/ANN/.

#### Search types


   * `priority`: visits cells in increasing order of distance from the query point, and hence, should converge more rapidly on the true nearest neighbour, but `standard` is usually faster for exact searches.
   * `radius`: only searches for neighbours within a specified radius of the point. If there are no neighbours then `nn_idx` will contain 0 and `nn_dists` will contain 1.340781e+154 for that point.

### Parameters
   *  **data**: *array_like*
      * M-D `np.matrix`, where each of the M rows is a point
      * M-D `np.ndarray`, where `D == 1` or `None`.
      
 
   *  **query**:  *array_like*, *optional*
   
      points that will be queried against data.
      
       * N-D `np.matrix`
       * N-D `np.array`, where `D == 1` or `None`
        
      `query.shape[1]` must `== data.shape[1]`. if `None` (default), `query == data`


   *  **k**:  *float*, *int*, *optional*
   
      The maximum number of nearest neighbours to compute. if `None` (default), k is set to `data.shape[0]` or 10, whichever smaller.


   *  **treetype**: *str*, *optional*
   
      Options:
        
       * `'kd'`: standard kd tree
       * `'bd'`: bd (box-decomposition, AMNSW98) tree which may perform better for larger point sets
        
      default is `'kd'`

   *   **searchtype**: *str*, *optional*
   
      Options:
      
       * `'standard'`
       * `'priority'`
       * `'radius'`
        
       See above for more detail. default is `'standard'`.

   *   **radius**: *float*, *int*, *optional*
   
       Radius of search for `searchtype='radius'`. default is `0.0`.

   *   **eps**: *float*, *int*, *optional*
     
       error bound. default of `0.0` implies exact nearest neighbour search.

### Return

   *   **NN2Results**
   
       Object of class `NN2Results` with two attributes: `nn_idx` and `nn_dists`.
       
       * `nn_idx`: A N-k integer `np.matrix` returning the near neighbour indices.
       * `nn_dists`: A N-k `np.matrix` returning the near neighbour Euclidean distances.

### Example

    > results = pyann.nn2(np.matrix([[1, 0],
                                    [2, 0]]),
                                    
                          np.matrix([[1.01, 0],
                                     [3,    0],
                                     [4.0, 0]]),
                          k=1)
    > results.nn_idx
    
    matrix([[1],
            [2],
            [2]])
    
    > results.nn_dists
    
    matrix([[0.01],
            [1.  ],
            [2.  ]])

