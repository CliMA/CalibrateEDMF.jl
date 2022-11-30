#!/bin/bash

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=50G   # memory per CPU core
#SBATCH -J "sct_init"   # job name

#julia package management
module load julia/1.7.3 hdf5/1.10.1 netcdf-c/4.6.1

config=${1?Error: no config file given}
job_id=${2?Error: no job ID given}

julia --project -C skylake-avx512 -JCEDMF.so init.jl --config $config --job_id $job_id && (
  echo sysimage loaded successfully
) || (
  julia --project init.jl --config $config --job_id $job_id
)

echo 'Ensemble initialized for calibration.'
