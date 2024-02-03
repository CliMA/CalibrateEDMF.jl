#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "up_deps"   # job name

module load julia/1.10.0 openmpi/4.1.5

julia --project .dev/up_deps.jl
julia --project=.dev .dev/climaformat.jl .

echo "Dependencies of all manifests updated."
