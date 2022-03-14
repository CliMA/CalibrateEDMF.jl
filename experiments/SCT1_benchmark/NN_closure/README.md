## Benchmark Results for Local Fully-Connected NN Closure

Results are given for the same architecture and different scaling parameters and regularizations.

#### 1/z scaling, regularized, 140 particles (`inv_z_l2_reg_0.1`)

Results after 43 iterations:
- MSE_full_mean (train) : ~ 3.01
- MSE_full_mean (validation) : ~ 4.78

- MSE_full_nn_mean (train) : ~ 3.01
- MSE_full_nn_mean (validation) : ~ 4.78

#### 1/z scaling, no regularization, 50 particles (`inv_z_e50`)

Results after 50 iterations:

- MSE_full_nn_mean (train) : ~ 3.08
- MSE_full_nn_mean (validation) : ~ 4.84

#### no scaling, no regularization, 50 particles (`none_e50`)

Results after 50 iterations:

- MSE_full_nn_mean (train) : ~ 2.95
- MSE_full_nn_mean (validation) : ~ 5.33

#### b/w scaling, no regularization, 50 particles (`buoy_vel_e50`)

Results after 50 iterations:

- MSE_full_nn_mean (train) : ~ 3.11
- MSE_full_nn_mean (validation) : ~ 4.65

# Details
  - NN Architecture: (6, 5, 4, 2) # (#inputs, #neurons in L1, #neurons in L2, #outputs)
  - NN parameters + turbulent entrainment coefficient -> 70 parameters
  - `TurbulenceConvection.jl v0.14.1` for `inv_z_l2_reg_0.1`
  - `TurbulenceConvection.jl v0.16.0` for `inv_z_e50`, `none_e50`, `buoy_vel_e50`