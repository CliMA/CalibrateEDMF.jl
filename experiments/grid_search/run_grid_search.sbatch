#!/bin/bash

#Submit this script with: sbatch calibrate_script

#SBATCH --time=02:00:00   # walltime
#SBATCH --ntasks=10  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "grd_src"   # job name
#SBATCH --output=slurm_grid_search_%j.out

config=${1?Error: no config file given}
# sim type should be 'reference' or 'validation'. Defaults to 'reference'
sim_type=${2:-reference}

module purge
module load julia/1.7.0 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
julia --project -p 10 grid_search.jl --config $config --sim_type $sim_type
