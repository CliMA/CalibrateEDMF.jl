## Training the TurbulenceConvection.jl EDMF with PyCLES data: An example

Authors: Ignacio Lopez-Gomez, Haakon Ludvig Langeland Ervik, Costa Christopoulos, Yair Cohen, Zhaoyi Shen

Last updated : March 2022

This is a pipeline example for the parameter calibration of a single-column EDMF model implemented in `TurbulenceConvection.jl` using PyCLES data as a reference. Calibration is performed using the `EnsembleKalmanProcesses.jl` package. The pipeline also enables calibration in a "perfect model" setting, that is, using `TurbulenceConvection.jl` as ground truth instead of LES data.

The configuration of the calibration process is centralized in `config.jl`, which stores details of the configuration in a hierarchichal dictionary structure. Different experiments sharing the majority of configuration details can be implemented rapidly dispatching on any of the subdictionaries by experiment type. This is shown within this example through the dispatch on `get_reference_config()`, which allows calibration of a `Bomex` or an `LesDrivenScm` simulation. The default is the calibration of a `Bomex` reference.

Two parallelized pipelines are able to ingest the `config.jl` file and perform the calibration, one using Julia's `pmap()` and another one using parallel bash calls through a SLURM HPC manager. Parallelization is performed over ensemble members. If the ensemble members have to perform several evaluations of `TurbulenceConvection.jl`, these are performed serially. A fully parallel implementation (`global_parallel`) that parallelizes across all `TurbulenceConvection.jl` is included in `experiments/SCT1_benchmark`. For calibration processes using only a few `TurbulenceConvection.jl` as data, the two pipelines presented here should suffice.

# How to run: Julia's pmap() pipeline

Within the `scm_pycles_pipeline` directory (i.e. this example), navigate to `julia_parallel`.

If you are on the Caltech Central Cluster, you can run the project by adding it to the schedule:

  >> sbatch calibrate_script

Otherwise run locally, e.g.:

>> sh calibrate_script

# How to run: SLURM HPC pipeline

Within the `scm_pycles_pipeline` directory (i.e. this example), navigate to `hpc_parallel`. The master bash script that starts the calibration process is `ekp_calibration.sbatch`, which takes as an argument the path to the config file.

If you are on the Caltech Central Cluster, you can run the project by adding it to the schedule:

  >> sbatch ekp_calibration.sbatch ../config.jl

# Output

While the simulations run, the results are dumped to a directory named *results_...* after every iteration of the calibration algorithm. Output of interest is stored in netCDF format, in `Diagnostics.nc`.
