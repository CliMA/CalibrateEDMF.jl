# Diagnostics Tools
A general set of tools for analyzing `CalibrateEDMF.jl` output and running `TurbulenceConvection.jl` with optimal parameters.

## DiagnosticsTools.jl
Utility functions to determine optimal parameters and mse values from completed calibration runs.
### Usage 
```
u_names, u = optimal_parameters(<PATH TO "Diagnostics.nc">; method = "best_nn_particle_mean", metric = "mse_full")
```
Given path to `Diagnostics.nc` file, return optimal parameter set across all iterations and corresponding parameter names. Defaults to particle nearest to ensemble mean (nearest neighbor) for the iteration with minimal mean mse on the training set. See docstring or table below for more options.

```
optimal_mse_train, optimal_mse_val = optimal_mse(<PATH TO "Diagnostics.nc">; method = "best_nn_particle_mean", metric = "mse_full")
```
Given path to `Diagnostics.nc` file, return minimum mse across iterations. Use in concert with `optimal_parameters` to find the mse associated with a set of optimal parameters. Defaults to returning training and validation mse of particle nearest to ensemble mean for the iteration with minimal mse on the training set.

##### Optimal Parameter Methods
| `method` | Description |
| ------- | -------|
| best_nn_particle_mean | Returns parameters for particle nearest to ensemble mean for the iteration with lowest mean training (`metric` = "mse_full") or validation (`metric` = "val_mse_full") mse. |
| best_particle | Returns parameters for particle with lowest mse on training (`metric` = "mse_full") or validation (`metric` = "val_mse_full") set. |

##### Optimal Parameter Metrics
| `metric` | Description |
| ------- | -------|
| mse_full | Find minimal mse of training set or mini-batch subset (`reference` in the calibration config.) |
| val_mse_full | Find minimal mse of validation set or mini-batch subset (`validation` in the calibration config.) |

Note that using mini-batching will result in an mse computed on a subset of the full training or validation dataset, so it is advantageous to evaluate (`metric` = val_mse_full) on a validation set without batching if batching is used for the training set. 

## TCRunner.jl
A command line script for running `TurbulenceConvection.jl` with a set of optimal parameters determined from `CalibrateEDMF.jl` on various datasets (`--run_set`). Runs validation cases from CEDMF config in `--results_dir` with optimal parameters by default. Ideally, calibration hyperparmeters and variations of the EDMF model are chosen using results on the validation set and offline performance is evaluated on the test set. Available `run_set`s:
| `run_set` | Description |
| ------- | -------|
| reference | Run TC with optimal parameters on all cases in the training set (`reference` in the calibration config.) |
| validation | Run TC with optimal parameters on all cases in the validation set (`validation` in the calibration config.) |
| test |  Run TC with optimal parameters on all cases in a newly-defined test set, (cases defined in `get_reference_config(::ScmTest)` in the config file defined at `--run_set_config`). See `tools/test_set_config.jl` for an example.|

### Usage 
```
julia --project TCRunner.jl --results_dir=<CEDMF OUTPUT DIR> --tc_output_dir=<DIR TO STORE TC OUTPUTS>
```
or use optional arguments for determining optimal parameters. For example, to run TC using parameters of the particle with lowest mse on the validation set:
```
julia --project TCRunner.jl --results_dir=<CEDMF OUTPUT DIR> --tc_output_dir=<DIR TO STORE TC OUTPUTS> --method="best_particle" --metric="val_mse_full"
```

