#!/bin/bash

#SBATCH --time=1:30:00   # walltime
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH -J "sys_im"   # job name

module load julia/1.9.3 openmpi/4.1.5

julia --project create_tc_so.jl

echo "System image creation/check finished."
