# Getting started

The main purpose of `CalibrateEDMF.jl` is to provide a user-friendly way to calibrate atmospheric turbulence and convection parameterizations within [TurbulenceConvection.jl](https://github.com/CliMA/TurbulenceConvection.jl) using gradient-free ensemble Kalman methods.

All user directives are specified through a `config.jl` file. `config.jl` is a Julia-readable collection of nested dictionaries that define all aspects of the calibration process, including:

 - The data used for training and online validation,
 - Inverse problem and optimization regularization,
 - The algorithm used for calibration (Ensemble Kalman Inversion, Unscented Kalman Inversion, ...) and its hyperparameters,
 - Parameter bounds and prior distributions.

Several `config.jl` example files are included in the `experiments/` directory of the package. The [Pipeline.jl](https://clima.github.io/CalibrateEDMF.jl/dev/API/Pipeline/) module implements the high-level functions called during a training session, from `config`-parsing to diagnostic output directives.

CalibrateEDMF.jl implements several calibration pipelines, designed to exploit available parallelisms of the computing station at the user's disposal.

## Julia's pmap() pipeline

For personal computers, parallelism is implemented through Julia's `pmap()` function.

### How to run (`pmap()` pipeline)

Navigate to `experiments/scm_pycles_pipeline/julia_parallel` directory. The script `calibrate.jl` implements the workflow of a training session that leverages Julia's `pmap()` to perform parallel `TurbulenceConvection.jl` evaluations within each training step. Evaluations are parallelized across ensemble members. 

If you are on the Caltech Central Cluster, you can run this pipeline by adding it to the SLURM schedule:

```
> sbatch calibrate_script ../config.jl
```

Otherwise run locally, e.g.:

```
> sh calibrate_script ../config.jl
```

Two parallelized pipelines are able to ingest the `config.jl` file and perform the calibration, one using Julia's `pmap()` and another one using parallel bash calls through a SLURM HPC manager. Parallelization is performed over ensemble members. If the ensemble members have to perform several evaluations of `TurbulenceConvection.jl`, these are performed serially. A fully parallel implementation (`global_parallel`) that parallelizes across all `TurbulenceConvection.jl` is included in `experiments/SCT1_benchmark`.

## SLURM HPC pipeline

For users with access to a high-performance computing (HPC) cluster, the `hpc_parallel` may be more efficient. The HPC pipeline parallelizes jobs using different Julia sessions per ensemble member and iteration. Due to the just-in-time compilation nature of Julia, this would normally require compiling the source code again for every new HPC node requested. In order to reduce the compilation overhead, this pipeline builds a system image of `EnsembleKalmanProcesses.jl`, `TurbulenceConvection.jl`, `CalibrateEDMF.jl` and all the functions called in the `CalibrateEDMF.jl` unit test suite. This system image uses a *frozen* version of the source code, so it must be re-generated every time any of the precompiled packages is updated or modified by the user.

### How to run (HPC pipeline)

Navigate to `experiments/scm_pycles_pipeline/hpc_parallel` directory. The master bash script that starts the calibration process is `ekp_calibration.sbatch`, which takes as an argument the path to the config file.

If you are on the Caltech Central Cluster, you can run the project by adding it to the schedule:

```
> sbatch ekp_calibration.sbatch ../config.jl
```

## Output of the training pipeline

While the simulations run, the results are dumped to a directory named *results_...* after every iteration of the calibration algorithm. Output of interest is stored in netCDF format, in `Diagnostics.nc`. To learn more about the diagnostics suite, see the [Diagnostics.jl](https://clima.github.io/CalibrateEDMF.jl/dev/API/Diagnostics/) module.

