#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "sct_run"   # job name

module load julia/1.10.1
iteration_=${1?Error: no iteration given}
job_id=${2?Error: no job ID given}
n_proc_scm=${3?Error: number of processes not given}

run_num=${SLURM_ARRAY_TASK_ID}
job_dir=$(head $job_id".txt" | tail -1)
version=$(head -"$run_num" $job_dir/"versions_"$iteration_".txt" | tail -1)

julia --project -C skylake-avx512 -JCEDMF.so -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train" && (
  echo sysimage loaded successfully
) || (
  julia --project -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train"
) &
P1=$!
julia --project -C skylake-avx512 -JCEDMF.so -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation" && (
  echo sysimage loaded successfully
) || (
  julia --project -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation"
) &
P2=$!
wait $P1 $P2
echo "SCM simulation for ${version} in iteration ${iteration_} finished"
