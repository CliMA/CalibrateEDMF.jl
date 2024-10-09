#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "sct_cont"   # job name

module load julia/1.9.3 openmpi/4.1.5
iteration_=${1?Error: no iteration given}
job_id=${2?Error: no job ID given}
job_dir=$(head $job_id".txt" | tail -1)

echo $job_id
echo $PWD

thisdir=${3?Error: Working directory not given}

# julia --project -C skylake-avx512 -JCEDMF.so step_ekp.jl --iteration $iteration_ --job_dir $job_dir && (
#   echo sysimage loaded successfully
# ) || (
#   julia --project step_ekp.jl --iteration $iteration_ --job_dir $job_dir
# )

julia --project $thisdir/step_ekp.jl --iteration $iteration_ --job_dir $job_dir # ignore system image for now (according to costa)


echo "Ensemble at iteration ${iteration_} stepped forward."
