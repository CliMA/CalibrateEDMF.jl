#!/bin/bash

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=12G   # memory per CPU core
#SBATCH -J "precond"   # job name

export MODULEPATH="/groups/esm/modules:$MODULEPATH"

module load julia/1.10.3
iteration_=${1?Error: no iteration given}
job_id=${2?Error: no job ID given}

run_num=${SLURM_ARRAY_TASK_ID}
job_dir=$(head $job_id".txt" | tail -1)
version=$(head -"$run_num" $job_dir/"versions_"$iteration_".txt" | tail -1)

julia --project precondition_prior.jl --version $version --job_dir $job_dir

echo "Preconditioning for ${version} finished."

