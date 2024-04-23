#!/bin/bash

#Submit this script with: sbatch calibrate_script

#SBATCH --time=02:00:00   # walltime
#SBATCH --ntasks=10  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "ekp_bomex"   # job name
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH --output=slurm_julia_par_%j.out

config=${1?Error: no config file given}

module purge
module load julia/1.10.1
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

julia --project -p 10 calibrate.jl --config $config
