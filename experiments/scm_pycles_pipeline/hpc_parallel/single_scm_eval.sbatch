#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=2       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "scm_run"   # job name

module load julia/1.10.0 openmpi/4.1.5
iteration_=${1?Error: no iteration given}
job_id=${2?Error: no job ID given}

run_num=${SLURM_ARRAY_TASK_ID}
job_dir=$(head $job_id".txt" | tail -1)
version=$(head -"$run_num" $job_dir/"versions_"$iteration_".txt" | tail -1)

julia --project -C skylake-avx512 -JCEDMF.so single_scm_eval.jl --version $version --job_dir $job_dir --mode "train" && (
  echo sysimage loaded successfully
) || (
  julia --project single_scm_eval.jl --version $version --job_dir $job_dir --mode "train"
) &
P1=$!
julia --project -C skylake-avx512 -JCEDMF.so single_scm_eval.jl --version $version --job_dir $job_dir --mode "validation" && (
  echo sysimage loaded successfully
) || (
  julia --project single_scm_eval.jl --version $version --job_dir $job_dir --mode "validation"
) &
P2=$!
wait $P1 $P2
echo "SCM simulation for ${version} in iteration ${iteration_} finished"

