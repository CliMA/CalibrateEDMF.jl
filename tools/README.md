# Diagnostics Tools
A general set of tools for analyzing `CalibrateEDMF.jl` output and running `TurbulenceConvection.jl` with optimal parameters.

## DiagnosticsTools.jl
Utility functions to determine optimal parameters and mse values from completed calibration runs.
### Usage 
```
u_names, u = optimal_parameters(<PATH TO "Diagnostics.nc">; method = "best_nn_particle_mean", metric = "mse_full")
```
Given path to `Diagnostics.nc` file, return optimal parameter set across all iterations and corresponding parameter names. Defaults to particle nearest to ensemble mean (nearest neighbor) for the iteration with minimal mse on the training set. See docstring for more options.

```
optimal_mse_train, optimal_mse_val = optimal_mse(<PATH TO "Diagnostics.nc">; method = "best_nn_particle_mean", metric = "mse_full")
```
Given path to `Diagnostics.nc` file, return minimum mse across iterations. Use in concert with `optimal_parameters` to find the mse associated with a set of optimal parameters. Defaults to returning training and validation mse of particle nearest to ensemble mean for the iteration with minimal mse on the training set.

## TCRunner.jl
A command line script for running `TurbulenceConvection.jl` with a set of optimal parameters determined from `CalibrateEDMF.jl`. Runs validation cases from config in `--results_dir` with optimal parameters by default.
### Usage 
```
julia --project TCRunner.jl --results_dir=<CEDMF OUTPUT DIR> --output_dir=<DIR TO STORE TC OUTPUTS>
```
or use optional arguments for determining optimal parameters. For example, to run TC using parameters of the particle with lowest mse on the validation set:
```
julia --project TCRunner.jl --results_dir=<CEDMF OUTPUT DIR> --output_dir=<DIR TO STORE TC OUTPUTS> --method="best_particle" --metric="val_mse_full"
```

