#!/bin/bash

#SBATCH --time=3:00:00   # walltime
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH -J "sys_im"   # job name

module load julia/1.7.3 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1

julia --project -C skylake-avx512 create_tc_so.jl

echo "System image creation/check finished."
