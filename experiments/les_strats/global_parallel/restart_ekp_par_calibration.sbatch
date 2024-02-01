#!/bin/bash

#SBATCH --time=0:20:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH -J "restart_call"   # job name

output_dir_rel=${1?Error: no output directory given}
output_dir=$(realpath $output_dir_rel)
config=${output_dir}/"config.jl"

# Job identifier
job_id=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 13 ; echo '')
# Get size of the ensemble from config, trim comments and whitespace
n=$(grep N_ens ${config} | cut -d=   -f2 | cut -d#   -f1 | xargs)
# Total umber of calibration iterations (previous+new), trim comments and whitespace
n_it=$(grep N_iter ${config} | cut -d=   -f2 | cut -d#   -f1 | xargs)
# Get last iteration
last_ekobj=$(ls -v ${output_dir}/ekobj_iter_* | tail -1)
last_iter=${last_ekobj%.*} # Remove trailing file format
last_iter=${last_iter##*_} # Remove leading filename
first_new_iter=$(($last_iter + 1))

# Number of parallel processes for SCM evaluation, split evenly between train and validation
n_proc_scm=10
echo "Restarting calibration with ${n} ensemble members and ${n_it} iterations (grand total)."

module purge
module load julia/1.10.0 openmpi/4.1.5

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

# run instantiate/precompile serial
julia --project -C skylake-avx512 -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project -C skylake-avx512 -e 'using Pkg; Pkg.precompile()'

for it in $(seq $first_new_iter 1 $n_it)
do
# Parallel runs of forward model
if [ "$it" = "$first_new_iter" ]; then
    # Generate system image
    id_so=$(sbatch --parsable -A esm sysimage.sbatch)
    # Restart calibration
    id_restart_ens=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_so restart.sbatch $output_dir $job_id)
    # Run ensemble of forward models
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_restart_ens --array=1-$n -n $n_proc_scm parallel_scm_eval.sbatch $it $job_id $n_proc_scm)
else
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ek_upd --array=1-$n -n $n_proc_scm parallel_scm_eval.sbatch $it $job_id $n_proc_scm)
fi
# Update ensemble
id_ek_upd=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ens_array --export=n=$n step_ekp.sbatch $it $job_id)
done

