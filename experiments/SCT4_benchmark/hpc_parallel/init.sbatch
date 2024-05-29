#!/bin/bash

#SBATCH --time=3:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "ekp_init"   # job name
#SBATCH --exclude=hpc-22-13,hpc-22-08

export MODULEPATH="/groups/esm/modules:$MODULEPATH"

#julia package management
module load julia/1.10.3
export CLIMACOMMS_DEVICE=CPU

config=${1?Error: no config file given}
job_id=${2?Error: no job ID given}

julia --project init.jl --config $config --job_id $job_id


echo 'Ensemble initialized for calibration.'
