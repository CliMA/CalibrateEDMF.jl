#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "restart"   # job name

module load julia/1.9.3 openmpi/4.1.5

output_dir=${1?Error: no output directory given}
job_id=${2?Error: no job ID given}

thisdir=/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Calibrate_exponential_T_scaling/global_parallel/ # can't figure out how to fool proof make this work w/ HPC even on interactive node which has a separate slurm id... this makes it however, not at all transporatable to other machines...

# julia --project -C skylake-avx512 -JCEDMF.so restart.jl --output_dir $output_dir --job_id $job_id && (
#   echo sysimage loaded successfully
# ) || (
#   julia --project restart.jl --output_dir $output_dir --job_id $job_id
# )

julia --project $thisdir/restart.jl --output_dir $output_dir --job_id $job_id # ignore system image for now (according to costa)


echo "Calibration process at ${output_dir} restarted."