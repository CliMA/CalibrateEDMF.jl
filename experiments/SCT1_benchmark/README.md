## Stratocumulus-to-Cumulus Transition Benchmark 1 (SCT1)

Authors: Ignacio Lopez-Gomez

Last updated : March 2022

This directory contains a `config.jl` file defining the SCT1 benchmark, along with files enabling efficient calibration in parallel in the Caltech central cluster. The SCT1 benchmark is composed of:

- A training set of 5 AMIP configurations, spanning locations from the California coast to Hawai'i. The simulations in the training set use as forcing conditions the October 2004-2008 climatology.

- A validation set of 5 AMIP4K configurations, spanning locations from the coast of Peru to the tropics. The simulations in the validation set use as forcing conditions the July 2004-2008 climatology.

This benchmark contains few simulations for tractability. It is meant to be a minimally working benchmark of subtropical cloud climatologies (training) and their evolution under a strong climate change scenario (validation).

# How to run: SLURM HPC pipeline

Navigate to `global_parallel`. The master bash script that starts the calibration process is `ekp_par_calibration.sbatch`, which takes as an argument the path to the config file.

If you are on the Caltech Central Cluster, you can run the project by adding it to the schedule:

  >> sbatch ekp_par_calibration.sbatch ../config.jl

The HPC pipeline parallelizes jobs using different Julia sessions per ensemble member and iteration. Due to the just-in-time compilation nature of Julia, this requires compiling the source code again for every new HPC node requested. In order to reduce the compilation overhead, this pipeline builds a system image of `EnsembleKalmanProcesses.jl`, `TurbulenceConvection.jl`, `CalibrateEDMF.jl` and all the functions called in the `CalibrateEDMF.jl` test suite. **This system image uses a *frozen* version of the source code, so it must be re-generated every time any of the precompiled packages is updated or modified by the user**.

# Output

While the simulations run, the results are dumped to a directory named *results_...* after every iteration of the calibration algorithm. The diagnostics are stored in NetCDF4 format in `Diagnostics.nc`, within the results folder. NetCDF files may be processed using python or julia. You may also access a human-readable version of the netCDF file in linux using `ncdump`:

  >> ncdump Diagnostics.nc > name_of_human_readable_file

# Benchmarking

Results for the `mse_full_nn_mean` and the spread (from `mse_full_var`) are included in `moisture_deficit_closure`. When using EKI, `mse_full_nn_mean` should be equivalent to `mse_full_mean` or `mse_full_min` after >10 iterations due to ensemble collapse. This reference calibrates 7 entrainment-related parameters from Cohen et al (2020). The evolution of the learnable parameters is also included in the directory.

Results for the moisture deficit closure after 50 iterations:

- MSE_full_nn_mean (train) : ~ 3.24
- MSE_full_nn_mean (validation) : ~ 4.7
