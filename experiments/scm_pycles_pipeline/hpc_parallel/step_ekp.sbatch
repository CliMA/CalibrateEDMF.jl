#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "ekp_cont"   # job name

module load julia/1.10.1
iteration_=${1?Error: no iteration given}
job_id=${2?Error: no job ID given}
job_dir=$(head $job_id".txt" | tail -1)

julia --project -C skylake-avx512 -JCEDMF.so step_ekp.jl --iteration $iteration_ --job_dir $job_dir && (
  echo sysimage loaded successfully
) || (
  julia --project step_ekp.jl --iteration $iteration_ --job_dir $job_dir
)

echo "Ensemble at iteration ${iteration_} stepped forward."
