#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "ens_init"   # job name

#julia package management
module load julia/1.6.0 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

n_ens=${1?Error: no ensemble size given}
job_id=${2?Error: no job ID given}

julia --project init.jl --n_ens $n_ens --job_id $job_id
echo 'Ensemble initialized for ensemble maps.'
