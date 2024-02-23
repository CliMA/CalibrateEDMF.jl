#!/bin/bash

#SBATCH --time=2:00:00    # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=15G # memory per CPU core (10G was what i was using, for 6 jobs parallel x6? idk...)  (reduced from 60G,... i think 10-15 should work, idk why it didnt work on the julia_parallel version... used to work when run separately... -- hopefully queues faster...
#SBATCH -J "sct_run"      # job name

# module load julia/1.9.3 openmpi/4.1.5
module load julia/1.9.3 mpich/4.0.0 # copied from other experiments


iteration_=${1?Error: no iteration given}
job_id=${2?Error: no job ID given}
n_proc_scm=${3?Error: number of processes not given}

run_num=${SLURM_ARRAY_TASK_ID}
job_dir=$(head $job_id".txt" | tail -1)
version=$(head -"$run_num" $job_dir/"versions_"$iteration_".txt" | tail -1)

echo $job_id
echo $version
echo $PWD
echo $job_dir

thisdir=${4?Error: Working directory not given}

# julia --project -C skylake-avx512 -JCEDMF.so -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train" && (
#   echo sysimage loaded successfully
# ) || (
#   julia --project -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train"
# ) &

julia --project -p $((n_proc_scm/2)) $thisdir/parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train" & # ignore system image for now (according to costa)
P1=$!
# julia --project -C skylake-avx512 -JCEDMF.so -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation" && (
#   echo sysimage loaded successfully
# ) || (
#   julia --project -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation"
# ) &

julia --project -p $((n_proc_scm/2)) $thisdir/parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation" & # ignore system image for now (according to costa)
P2=$!
wait $P1 $P2
echo "SCM simulation for ${version} in iteration ${iteration_} finished"
