#!/bin/bash

#Submit this script with: sbatch calibrate_script

#SBATCH --time=30:00:00   # walltime
#SBATCH --ntasks=25  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "r_grid"   # job name
#SBATCH --output=slurm_grid_search_%j.out

outdir=${1?Error: no output directory given}

module purge
module load julia/1.9.3 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
julia --project -p 25 grid_search.jl --outdir $outdir --mode restart
