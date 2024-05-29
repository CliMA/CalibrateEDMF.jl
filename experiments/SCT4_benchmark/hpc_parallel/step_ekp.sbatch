#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=15G   # memory per CPU core
#SBATCH -J "ekp_cont"   # job name
#SBATCH --exclude=hpc-22-13,hpc-22-08

export MODULEPATH="/groups/esm/modules:$MODULEPATH"

module load julia/1.10.3
export CLIMACOMMS_DEVICE=CPU

iteration_=${1?Error: no iteration given}
job_id=${2?Error: no job ID given}
job_dir=$(head $job_id".txt" | tail -1)


julia --project step_ekp.jl --iteration $iteration_ --job_dir $job_dir


echo "Ensemble at iteration ${iteration_} stepped forward."
