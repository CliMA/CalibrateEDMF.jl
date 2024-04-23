#!/bin/bash

#Submit this script with: sbatch TCRunner.sbatch

#SBATCH --time=06:00:00   # walltime
#SBATCH --ntasks=16  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "TCRunner"   # job name
#SBATCH --output=slurm_TCRunner_%j.out

module purge
module load julia/1.10.1
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

julia -p15 TCRunner.jl "$@"

echo finished
