#!/bin/bash

#SBATCH --time=3:00:00   # walltime
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH -J "sys_im"   # job name

module load julia/1.9.3 openmpi/4.1.5

thisdir=/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Calibrate_exponential_T_scaling/global_parallel/ # can't figure out how to fool proof make this work w/ HPC even on interactive node which has a separate slurm id... this makes it however, not at all transporatable to other machines...

julia --project -C skylake-avx512 $thisdir/create_tc_so.jl

echo "System image creation/check finished."
