#!/bin/bash

#SBATCH --time=0:20:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH -J "restart_call"   # job name

output_dir=${1?Error: no output directory given}
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
module load julia/1.7.0 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

# run instantiate/precompile serial
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project -e 'using Pkg; Pkg.precompile()'

# First call to calibrate.jl will create the ensemble files from the priors
id_restart_ens=$(sbatch --parsable -A esm restart.sbatch $output_dir $job_id)
for it in $(seq $first_new_iter 1 $n_it)
do
# Parallel runs of forward model
if [ "$it" = "$first_new_iter" ]; then
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_restart_ens --array=1-$n -n $n_proc_scm parallel_scm_eval.sbatch $it $job_id $n_proc_scm)
else
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ek_upd --array=1-$n -n $n_proc_scm parallel_scm_eval.sbatch $it $job_id $n_proc_scm)
fi
id_ek_upd=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ens_array --export=n=$n step_ekp.sbatch $it $job_id)
done

