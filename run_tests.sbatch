#!/bin/bash

#SBATCH --time=0:20:00   # walltime
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "tests"   # job name

module load julia/1.7.3

julia --project test/runtests.jl
