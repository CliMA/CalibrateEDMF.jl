## Benchmark Results for Local Fully-Connected NN Closure

Results are given for the same architecture and different scaling parameters and regularizations.

#### 1/z scaling, no NN regularization, regularized physical parameters (`inv_z_no_NN_reg`)

Results after 50 iterations:

###### 50 particles:

- MSE_full_nn_mean (train) : ~ (2, 4) (batching)
- MSE_full_nn_mean (validation) : ~ 3.41

# Details
  - NN Architecture: (6, 5, 4, 2) # (#inputs, #neurons in L1, #neurons in L2, #outputs)
  - NN parameters + physics-based empirical parameters -> 79 parameters
  - `TurbulenceConvection.jl v0.16.0`
