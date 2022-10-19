#!/bin/bash

#Submit this script with: sbatch calibrate_script

#SBATCH --time=30:00:00   # walltime
#SBATCH --ntasks=25  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "grid"   # job name
#SBATCH --output=slurm_grid_search_%j.out

config=${1?Error: no config file given}

module purge
module load julia/1.7.3 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
julia --project -p 25 grid_search.jl --config $config --mode new
