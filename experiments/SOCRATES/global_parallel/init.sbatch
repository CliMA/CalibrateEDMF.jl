#!/bin/bash

#SBATCH --time=2:00:00      # walltime (what times do we need for our runs)
#SBATCH --ntasks=1          # number of processor cores (i.e. tasks)
#SBATCH --nodes=1           # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH -J "sct_init"       # job name

#julia package management
# module load julia/1.9.3 openmpi/4.1.5
module load julia/1.10.0 mpich/4.0.0 # copied from other experiments

config=${1?Error: no config file given}
job_id=${2?Error: no job ID given}

# julia --project -C skylake-avx512 -JCEDMF.so init.jl --config $config --job_id $job_id && (
#   echo sysimage loaded successfully
# ) || (
#   julia --project init.jl --config $config --job_id $job_id
# )

thisdir=${3?Error: Working directory not given}

julia --project $thisdir/init.jl --config $config --job_id $job_id 

echo 'Ensemble initialized for calibration.'

