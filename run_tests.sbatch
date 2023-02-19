#!/bin/bash

#SBATCH --time=0:40:00   # walltime
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "tests"   # job name

module load julia/1.8.5

julia --project test/runtests.jl
