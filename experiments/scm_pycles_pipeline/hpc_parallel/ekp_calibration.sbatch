#!/bin/bash

#SBATCH --time=0:10:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH -J "ekp_call"   # job name

config_rel=${1?Error: no config file given}
config=$(realpath $config_rel)

# Job identifier
job_id=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 13 ; echo '')
# Get size of the ensemble from config, trim comments and whitespace
n=$(grep N_ens ${config} | cut -d=   -f2 | cut -d#   -f1 | xargs)
# Number of calibration iterations, trim comments and whitespace
n_it=$(grep N_iter ${config} | cut -d=   -f2 | cut -d#   -f1 | xargs)
echo "Initializing calibration with ${n} ensemble members and ${n_it} iterations."

module purge
module load julia/1.8.3 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

# run instantiate/precompile serial
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project -e 'using Pkg; Pkg.precompile()'

for it in $(seq 1 1 $n_it)
do
# Parallel runs of forward model
if [ "$it" = "1" ]; then
    # Generate system image
    id_so=$(sbatch --parsable -A esm sysimage.sbatch)
    # Initialize calibration
    id_init_ens=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_so init.sbatch $config $job_id)
    # Precondition parameters
    id_precond=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_init_ens --array=1-$n precondition_prior.sbatch $it $job_id)
    # Run ensemble of forward models
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_precond --array=1-$n single_scm_eval.sbatch $it $job_id)
else
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ek_upd --array=1-$n single_scm_eval.sbatch $it $job_id)
fi
# Update ensemble
id_ek_upd=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ens_array --export=n=$n step_ekp.sbatch $it $job_id)
done
