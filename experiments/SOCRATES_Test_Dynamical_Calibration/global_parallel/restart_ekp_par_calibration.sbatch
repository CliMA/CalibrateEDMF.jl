#!/bin/bash

#SBATCH --time=0:20:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH -J "restart_call"   # job name
#SBATCH -o /dev/null

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

echo "output_dir: ${output_dir}"
echo "job_id: ${job_id}"
echo "last iteration: ${last_iter}"
echo "last ekobj: ${last_ekobj}"
echo "first new iteration: ${first_new_iter}"

# Number of parallel processes for SCM evaluation, split evenly between train and validation
n_proc_scm=10
echo "Restarting calibration with ${n} ensemble members and ${n_it} iterations (grand total)."

module purge
module load julia/1.9.3 openmpi/4.1.5

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false
julia -e 'using Pkg; using MPIPreferences; use_system_binary()' # the new way to do this, see ` Warning: The JULIA_MPI_BINARY environment variable is no longer used to configure the MPI binary. Use MPIPreferences.use_system_binary() instead.`

# run instantiate/precompile serial
julia --project -C skylake-avx512 -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project -C skylake-avx512 -e 'using Pkg; Pkg.precompile()'

thisfile=/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Test_Dynamical_Calibration/global_parallel/restart_ekp_par_calibration.sbatch # can't figure out how to fool proof make this work w/ HPC even on interactive node which has a separate slurm id... this makes it however, not at all transporatable to other machines...
thisdir=$(dirname $thisfile) # go up to directory from filename...
logdir=~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Test_Dynamical_Calibration/global_parallel/slurm/

for it in $(seq $first_new_iter 1 $n_it)
do
# Parallel runs of forward model
if [ "$it" = "$first_new_iter" ]; then
    # Generate system image
    # id_so=$(sbatch -o $logdir/%x-%j.out --parsable -A esm $thisdir/sysimage.sbatch)
    # Restart calibration
    # id_restart_ens=$(sbatch -o $logdir/%x-%j.out --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_so $thisdir/restart.sbatch $output_dir $job_id)
    id_restart_ens=$(sbatch -o $logdir/%x-%j.out --parsable                             -A esm                             $thisdir/restart.sbatch $output_dir $job_id) # turn off system image for now
    # Run ensemble of forward models
    id_ens_array=$(sbatch -o $logdir/%x-%j.out --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_restart_ens --array=1-$n -n $n_proc_scm $thisdir/parallel_scm_eval.sbatch $it $job_id $n_proc_scm)
else
    id_ens_array=$(sbatch -o $logdir/%x-%j.out --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ek_upd --array=1-$n -n $n_proc_scm $thisdir/arallel_scm_eval.sbatch $it $job_id $n_proc_scm)
fi
# Update ensemble
id_ek_upd=$(sbatch -o $logdir/%x-%j.out --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ens_array --export=n=$n $thisdir/step_ekp.sbatch $it $job_id)
done

