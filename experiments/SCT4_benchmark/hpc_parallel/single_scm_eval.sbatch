#!/bin/bash

#SBATCH --time=3:00:00   # walltime
#SBATCH --ntasks=5       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "scm_run"   # job name
#SBATCH --exclude=hpc-22-13,hpc-22-08

export MODULEPATH="/groups/esm/modules:$MODULEPATH"

module load julia/1.10.3
export CLIMACOMMS_DEVICE=CPU

julia --project -e 'println("-------cpu inf0-------")'
julia --project -e 'println(Sys.CPU_NAME)'
julia --project -e 'println("Architecture: ", Sys.ARCH)'
julia --project -e 'println("Number of Logical CPU Cores: ", Sys.CPU_THREADS)'
julia --project -e 'println("Julia version: ", VERSION)'
julia --project -e 'println("Load path: ", LOAD_PATH)'
julia --project -e 'println("Depot path: ", DEPOT_PATH)'


# julia --project -e 'using Pkg; Pkg.precompile()'

iteration_=${1?Error: no iteration given}
job_id=${2?Error: no job ID given}

run_num=${SLURM_ARRAY_TASK_ID}
job_dir=$(head $job_id".txt" | tail -1)
version=$(head -"$run_num" $job_dir/"versions_"$iteration_".txt" | tail -1)


julia --project single_scm_eval.jl --version $version --job_dir $job_dir --mode "train" & P1=$!
julia --project single_scm_eval.jl --version $version --job_dir $job_dir --mode "validation" & P2=$!

wait $P1 $P2
echo "SCM simulation for ${version} in iteration ${iteration_} finished"

