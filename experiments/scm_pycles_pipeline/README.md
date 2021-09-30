## Training the TurbulenceConvection.jl EDMF with PyCLES data: An example

Authors: Yair Cohen, Haakon Ludvig Langeland Ervik, Ignacio Lopez-Gomez, Zhaoyi Shen

Last updated : September 2021

This example script details the calibration of parameters for a single-column EDMF model implemented in `TurbulenceConvection.jl` using PyCLES data as a reference. The script also enables calibration in the "perfect model" setting.

# How to run

Within the `scm_pycles_pipeline` directory (i.e. this example), open `calibrate.jl`.

The `ReferenceModel` allows you to select where to fetch input data. The `outdir_root` variable determines the base directory for output.

If you are on the Caltech Central Cluster, you can run the project by adding it to the schedule:

  >> sbatch calibrate_script

Otherwise run locally, e.g.:

>> sh calibrate_script

After the simulations are done, the results will be stored in a directory named *results_...*
Output of interest is stored in `.jld2` format, along with a figure of the evolution of the parameter ensemble and its error.
