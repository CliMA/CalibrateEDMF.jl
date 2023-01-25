#!/bin/bash

#Submit this script with: sbatch calibrate_script

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=25  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "loss"   # job name
#SBATCH --output=slurm_loss_map_%j.out

sim_dir=${1?Error: no simulation path given}

module purge
module load julia/1.8.3 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
julia --project -p 25 loss_map.jl --sim_dir $sim_dir
