## Training the TurbulenceConvection.jl EDMF with PyCLES data: An example

Authors: Yair Cohen, Haakon Ludvig Langeland Ervik, Ignacio Lopez-Gomez, Zhaoyi Shen

Last updated : September 2021

This example script details the calibration of parameters for a single-column EDMF model implemented in `TurbulenceConvection.jl` using PyCLES data as a reference. The script also enables calibration in the "perfect model" setting.

To perform simulations, first we need to clone this repo and the `TurbulenceConvection.jl` repo from GitHub.

  >> git clone https://github.com/CliMA/CalibrateEDMF.jl.git

  >> git clone https://github.com/CliMA/TurbulenceConvection.jl 

Before calibration, we need to compile the project. Note that each experiment in CalibrateEDMF.jl is its own project, so we will have to perform compilation within each example in the `experiments` directory. For the time being `TurbulenceConvection.jl` is not a published package, so the dependencies are added manually by

>> julia --project

>> julia> ]

>> pkg> dev path/to/TurbulenceConvection.jl

>> pkg> instantiate

Then, navigate into the `scm_pycles_pipeline` directory (i.e. this example), and open `calibrate.jl`.

The `ReferenceModel` allow you to select where to fetch input data. The `outdir_root` variable determines the base directory for output.

If you are on the Caltech Central Cluster, you run the project by adding it to the schedule:

  >> sbatch calibrate_script

otherwise run locally, e.g.:

>> sh calibrate_script

After the simulations are done, the results will be stored in a directory named *results_...*
Output of interest is stored in jld2 format, along with a figure of the evolution of the parameter ensemble and its error.
