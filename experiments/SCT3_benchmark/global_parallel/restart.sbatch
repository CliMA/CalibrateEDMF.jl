#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "restart"   # job name

export MODULEPATH="/groups/esm/modules:$MODULEPATH"

module load julia/1.10.1

output_dir=${1?Error: no output directory given}
job_id=${2?Error: no job ID given}

julia --project restart.jl --output_dir $output_dir --job_id $job_id

echo "Calibration process at ${output_dir} restarted."