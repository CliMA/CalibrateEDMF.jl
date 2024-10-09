#!/bin/bash

#SBATCH --time=2:00:00    # walltime
#SBATCH --ntasks=1        # number of processor cores (i.e. tasks) # increase number of tasks ...
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=60G # memory per CPU core (10G was what i was using, for 6 jobs parallel x6? idk...)
#SBATCH -J "precond"      # job name

module load julia/1.9.3 openmpi/4.1.5
iteration_=${1?Error: no iteration given}
job_id=${2?Error: no job ID given}

run_num=${SLURM_ARRAY_TASK_ID}
job_dir=$(head $job_id".txt" | tail -1)
version=$(head -"$run_num" $job_dir/"versions_"$iteration_".txt" | tail -1)

echo $job_id
echo $version
echo $PWD

thisdir=/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Test_Dynamical_Calibration/global_parallel/ # can't figure out how to fool proof make this work w/ HPC even on interactive node which has a separate slurm id... this makes it however, not at all transporatable to other machines...

# julia --project -C skylake-avx512 -JCEDMF.so precondition_prior.jl --version $version --job_dir $job_dir && (
#   echo sysimage loaded successfully
# ) || (
#   julia --project precondition_prior.jl --version $version --job_dir $job_dir
# )

julia --project $thisdir/precondition_prior.jl --version $version --job_dir $job_dir # ignore system image for now (according to costa)

echo "Preconditioning for ${version} finished."

