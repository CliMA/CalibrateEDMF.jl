## Benchmark Results for Local Fully-Connected NN Closure

Results are given for the same architecture and different scaling parameters and regularizations.

#### 1/z scaling, regularized, 140 particles (`inv_z_l2_reg_0.1`)

Results after 43 iterations:
- MSE_full_mean (train) : ~ 3.01
- MSE_full_mean (validation) : ~ 4.78

- MSE_full_nn_mean (train) : ~ 3.01
- MSE_full_nn_mean (validation) : ~ 4.78

#### 1/z scaling, no regularization (`inv_z_no_reg`)

Results after 50 iterations:

###### 50 particles:

- MSE_full_nn_mean (train) : ~ 3.08
- MSE_full_nn_mean (validation) : ~ 4.84

###### 100 particles:

- MSE_full_nn_mean (train) : ~ 3.00
- MSE_full_nn_mean (validation) : ~ 4.75


#### no scaling, no regularization, (`none_no_reg`)

Results after 50 iterations:

###### 50 particles:

- MSE_full_nn_mean (train) : ~ 2.95 (unstable posterior)
- MSE_full_nn_mean (validation) : ~ 5.33 (unstable posterior)

###### 100 particles:

- MSE_full_nn_mean (train) : ~ 2.86 (unstable posterior)
- MSE_full_nn_mean (validation) : ~ 4.70 (unstable posterior)


#### b/w scaling, no regularization (`buoy_vel_no_reg`)

Results after 50 iterations:

###### 50 particles:

- MSE_full_nn_mean (train) : ~ 3.11
- MSE_full_nn_mean (validation) : ~ 4.65

###### 100 particles:

- MSE_full_nn_mean (train) : ~ 2.90 (unstable posterior)
- MSE_full_nn_mean (validation) : ~ 4.70 (unstable posterior)

# Details
  - NN Architecture: (6, 5, 4, 2) # (#inputs, #neurons in L1, #neurons in L2, #outputs)
  - NN parameters + turbulent entrainment coefficient -> 70 parameters
  - `TurbulenceConvection.jl v0.14.1` for `inv_z_l2_reg_0.1`
  - `TurbulenceConvection.jl v0.16.0` for `inv_z_no_reg`, `none_no_reg`, `buoy_vel_no_reg`
  - For this benchmark and without regularization, the `inv_z` seems to stabilize training,
    compared to the rest of scalings.
