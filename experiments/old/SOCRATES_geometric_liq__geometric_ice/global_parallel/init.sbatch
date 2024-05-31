#!/bin/bash

#SBATCH --time=2:00:00      # walltime (what times do we need for our runs)
#SBATCH --ntasks=1          # number of processor cores (i.e. tasks)
#SBATCH --nodes=1           # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH -J "sct_init"       # job name

#julia package management
module load julia/1.9.3 openmpi/4.1.5

config=${1?Error: no config file given}
job_id=${2?Error: no job ID given}

# julia --project -C skylake-avx512 -JCEDMF.so init.jl --config $config --job_id $job_id && (
#   echo sysimage loaded successfully
# ) || (
#   julia --project init.jl --config $config --job_id $job_id
# )

thisdir=/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Calibrate_exponential_T_scaling/global_parallel/ # can't figure out how to fool proof make this work w/ HPC even on interactive node which has a separate slurm id... this makes it however, not at all transporatable to other machines...

julia --project $thisdir/init.jl --config $config --job_id $job_id # ignore system image for now (according to costa)


echo 'Ensemble initialized for calibration.'

