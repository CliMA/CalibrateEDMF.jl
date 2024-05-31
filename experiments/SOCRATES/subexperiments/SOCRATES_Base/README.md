## Stratocumulus-to-Cumulus Transition Benchmark 2 (SCT2)

Authors: Ignacio Lopez-Gomez

Last updated : March 2022

This directory contains a `config.jl` file defining the SCT2 benchmark, along with files enabling efficient calibration in parallel in the Caltech central cluster. The SCT2 benchmark is composed of:

- A training set of 60 AMIP configurations, spanning locations from the California coast to Hawai'i. The simulations in the training set use as forcing conditions the 2004-2008 climatology for January, April, July and October.

- A validation set of 5 AMIP4K configurations, spanning locations from the coast of Peru to the tropics. The simulations in the validation set use as forcing conditions the July 2004-2008 climatology.

This benchmark contains few validation simulations for tractability. For the same reason, it is recommended to keep the batch size small (the default is 5).

# How to run: SLURM HPC pipeline

Navigate to `global_parallel`. The master bash script that starts the calibration process is `ekp_par_calibration.sbatch`, which takes as an argument the path to the config file.

If you are on the Caltech Central Cluster, you can run the project by adding it to the schedule:

  >> sbatch ekp_par_calibration.sbatch ../config.jl

# Output

While the simulations run, the results are dumped to a directory named *results_...* after every iteration of the calibration algorithm. The diagnostics are stored in NetCDF4 format in `Diagnostics.nc`, within the results folder. NetCDF files may be processed using python or julia. You may also access a human-readable version of the netCDF file in linux using `ncdump`:

  >> ncdump Diagnostics.nc > name_of_human_readable_file
