#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "ekp_call"   # job name

# Job identifier
job_id=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 13 ; echo '')
# Size of the ensemble
n=5
# Number of EK iterations
n_it=4

module purge
module load julia/1.6.0 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

# run instantiate/precompile serial
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project -e 'using Pkg; Pkg.precompile()'

# First call to calibrate.jl will create the ensemble files from the priors
id_init_ens=$(sbatch --parsable -A esm init.sbatch $n $n_it $job_id)
for it in $(seq 1 1 $n_it)
do
# Parallel runs of forward model
if [ "$it" = "1" ]; then
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_init_ens --array=1-$n single_scm_eval.sbatch $it $job_id)
else
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ek_upd --array=1-$n single_scm_eval.sbatch $it $job_id)
fi
id_ek_upd=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ens_array --export=n=$n step_ekp.sbatch $it $job_id)
done

