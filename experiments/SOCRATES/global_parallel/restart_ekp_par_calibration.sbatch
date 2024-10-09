#!/bin/bash

#SBATCH --time=0:20:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH -J "restart_call"   # job name

output_dir_rel=${1?Error: no output directory given}
output_dir=$(realpath $output_dir_rel)
config=${output_dir}/"config.jl"

# check if fourth argument is given (use_expansion)
if [ -z "$2" ]; then
    use_expansion=false
else
    use_expansion=${2}
fi
echo "use_expansion: ${use_expansion}"

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
n_proc_scm=6
echo "Restarting calibration with ${n} ensemble members and ${n_it} iterations (grand total)."

# experiment_dir=${2?Error: Working directory not given}

config_dir=$(dirname $(dirname $(dirname $output_dir_rel)))/configs # go up from the output dir (original input w/ symlinks, not the realpath) to the configs dir (have to use dirname bc it doesnt resolve symlinks, realpath and `..` do )
echo $config_dir
subexperiment_dir=$(realpath $config_dir/../../../../../../) # go up from the config paths in Calibrate_and_Run/<calibration setup>/<dt string> <calibration vars>/calibrate/configs/ to the subexperiment directory
echo $subexperiment_dir
experiment_dir=$(realpath $subexperiment_dir/../../) # go up from the subexperiment directory to the experiment directory
echo $experiment_dir
thisfile=$experiment_dir/global_parallel/ekp_par_calibration.sbatch # can't figure out how to fool proof make this work w/ HPC even on interactive node which has a separate slurm id... this makes it however, not at all transporatable to other machines...
thisdir=$(dirname $thisfile) # go up to directory from filename...
logdir=$experiment_dir/global_parallel/slurm/




thisdir=$experiment_dir/global_parallel/ # can't figure out how to fool proof make this work w/ HPC even on interactive node which has a separate slurm id... this makes it however, not at all transporatable to other machines...
logdir=$experiment_dir/global_parallel/slurm/

cd $thisdir # should make everything work later, so that --project picks up the right location (esp since we're using symlinks) but on slurm this is bad since we should be in /var no? Alternative is to set --project everywhere (or maybe it's fine, only the calling script is in var or use -D option to set home directory)
# note, wherever you cd to is also where files will get put, e.g. ${job_id}.txt, but we don't know the output_dir here so maybe put it w/ slurm? idk...
cd $logdir # test to see if it still crashes -- hopefully it just puts ${job_id}.txt in the slurm directory and the path specs everywhere else work... -- is there any way to get the output directory? I guess you could read it from this file, move the job_id.txt file there then pass that path instead to the other files....



if $(grep -q "Red Hat" /etc/redhat-release);  then # copied from /groups/esm/slurm-buildkite/hooks/environment
    os_type="rhel9" # "We are on a RedHat 9 node"
else 
	os_type="centos7" # some of the expansion nodes still seem to be running centos7 for some strange reason? idk...
fi

if [ "$use_expansion" = false ] || [ "$os_type" = "centos7" ]; then # use_expansion is false or os_type is centos7 despite being on expansion node (idk why that is happening lol)

    module purge
    module load julia/1.10.0 mpich/4.0.0 # copied from other experiments

    export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
    # export JULIA_MPI_BINARY=system
    export JULIA_CUDA_USE_BINARYBUILDER=false
    # julia -e 'using Pkg; Pkg.add("MPIPreferences")' # the new way to do this, see ` Warning: The JULIA_MPI_BINARY environment variable is no longer used to configure the MPI binary. Use MPIPreferences.use_system_binary() instead.`
    julia -e 'using Pkg; using MPIPreferences; use_jll_binary()' # the new way to do this, see ` Warning: The JULIA_MPI_BINARY environment variable is no longer used to configure the MPI binary. Use MPIPreferences.use_system_binary() instead.`

    # run instantiate/precompile serial
    export JULIA_CPU_TARGET="generic;skylake-avx512,clone_all;skylake,clone_all;cascadelake,clone_all;icelake,clone_all;icelake-server,clone_all;broadwell,clone_all" # hopefully this sets the cpu_target such that we avoid repeated precompilation....
    julia --project -e 'using Pkg; Pkg.instantiate()' # no sysimage, hopefully this is compiled such that there's nothing to do lol.... (i'm not sure just doing this picks up changes in dev'd packages though, I'll have to look into that)
    
    # julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()' # no sysimage, no skylake-avx512
    # julia --project -e 'using Pkg; Pkg.precompile()'

    for it in $(seq $first_new_iter 1 $n_it)
    do
    echo $it

    # Parallel runs of forward model
    if [ "$it" = "$first_new_iter" ]; then
        # Generate system image
        # id_so=$(sbatch -o $logdir/%x-%j.out --parsable $thisdir/sysimage.sbatch)
        # Restart calibration
        # id_restart_ens=$(sbatch -o $logdir/%x-%j.out --parsable --kill-on-invalid-dep=yes --dependency=afterok:$id_so $thisdir/restart.sbatch $output_dir $job_id)
        id_restart_ens=$(sbatch -o $logdir/%x-%j.out --parsable   $thisdir/restart.sbatch $output_dir $job_id $thisdir) # turn off system image for now
        # Run ensemble of forward models
        id_ens_array=$(sbatch -o $logdir/%x-%A_%a.out --parsable --kill-on-invalid-dep=yes --dependency=afterok:$id_restart_ens --array=1-$n -n $n_proc_scm $thisdir/parallel_scm_eval.sbatch $it $job_id $n_proc_scm $thisdir)
    else
        id_ens_array=$(sbatch -o $logdir/%x-%A_%a.out --parsable --kill-on-invalid-dep=yes --dependency=afterok:$id_ek_upd --array=1-$n -n $n_proc_scm $thisdir/parallel_scm_eval.sbatch $it $job_id $n_proc_scm $thisdir)
    fi
    # Update ensemble
    id_ek_upd=$(sbatch -o $logdir/%x-%j.out --parsable --kill-on-invalid-dep=yes --dependency=afterok:$id_ens_array --export=n=$n $thisdir/step_ekp.sbatch $it $job_id $thisdir)
    done

else
    # cat /etc/os-release # just to be sure
    source /etc/profile.d/modules.sh # # module seems to not be defined when we call this on expansion? idk why... hopefully this does it automatically, see https://climate-machine.slack.com/archives/CKH8UUZHR/p1700178461357699?thread_ts=1700174744.568459&cid=CKH8UUZHR 
    # Try running on expansion cores
    module purge
    export MODULEPATH="/groups/esm/modules:$MODULEPATH" # see https://github.com/CliMA/ClimaModules, provides julia, julia-preferences, mpiwrapper
    export MODULEPATH="/central/software9/Modules/linux-rhel9-x86_64:/central/software9/Modules/linux-rhel9-broadwell:/central/software9/Modules/linux-rhel9-skylake_avx512:/central/software9/Modules/external/modulefiles:$MODULEPATH" # for some reason these aren't getting loaded ... idk why... it's some weird mix of expansion and regular modules  but only the regular centos7 are getting loaded by sbatch idk but not the rhel9 ones..., plus /central/software9/Modules/external/modulefiles for nsight_systems for climacommon
    # # module load climacommon
    # module load climacommon/2024_10_08 # avoid julia 1.11 for now # doesnt work bc of nsys
    # module load julia/1.10.1 
    # module load julia-preferences/2024_02_20
    # module load mpiwrapper/2024_02_27 # is this all we need for mpi? the code still complained about libmpi not being found but it still ran? idk...
    # module load climacommon
    module load climacommon/2024_10_08 # avoid julia 1.11 for now

    export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
    # export JULIA_MPI_BINARY=system # I think we don't need this anymore
    export JULIA_CUDA_USE_BINARYBUILDER=false
    # julia -e 'using Pkg; Pkg.add("MPIPreferences")' # the new way to do this, see ` Warning: The JULIA_MPI_BINARY environment variable is no longer used to configure the MPI binary. Use MPIPreferences.use_system_binary() instead.`
    # julia -e 'using Pkg; using MPIPreferences; use_system_binary()' # the new way to do this, see ` Warning: The JULIA_MPI_BINARY environment variable is no longer used to configure the MPI binary. Use MPIPreferences.use_system_binary() instead.`
    julia -e 'using Pkg; using MPIPreferences; use_jll_binary()' # maybe we're supposed to start using jll binary?

    # run instantiate/precompile serial
    export JULIA_CPU_TARGET="generic;skylake-avx512,clone_all;skylake,clone_all;cascadelake,clone_all;icelake,clone_all;icelake-server,clone_all;broadwell,clone_all" # hopefully this sets the cpu_target such that we avoid repeated precompilation....
    julia --project -e 'using Pkg; Pkg.instantiate()' # no sysimage, hopefully this is compiled such that there's nothing to do lol.... (i'm not sure just doing this picks up changes in dev'd packages though, I'll have to look into that)
    # julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()' # no sysimage, no skylake-avx512
    # julia --project -e 'using Pkg; Pkg.precompile()'

    for it in $(seq $first_new_iter 1 $n_it)
    do
    echo $it

    # Parallel runs of forward model
    if [ "$it" = "$first_new_iter" ]; then
        # Generate system image
        # id_so=$(sbatch -o $logdir/%x-%j.out --parsable --partition=expansion $thisdir/sysimage.sbatch)
        # Restart calibration
        # id_restart_ens=$(sbatch -o $logdir/%x-%j.out --parsable --partition=expansion --kill-on-invalid-dep=yes --dependency=afterok:$id_so $thisdir/restart.sbatch $output_dir $job_id)
        id_restart_ens=$(sbatch -o $logdir/%x-%j.out --parsable --partition=expansion $thisdir/restart.sbatch $output_dir $job_id $thisdir $use_expansion) # turn off system image for now
        # Run ensemble of forward models
        id_ens_array=$(sbatch -o $logdir/%x-%A_%a.out --parsable --partition=expansion --kill-on-invalid-dep=yes --dependency=afterok:$id_restart_ens --array=1-$n -n $n_proc_scm $thisdir/parallel_scm_eval.sbatch $it $job_id $n_proc_scm $thisdir $use_expansion)
    else
        id_ens_array=$(sbatch -o $logdir/%x-%A_%a.out --parsable --partition=expansion --kill-on-invalid-dep=yes --dependency=afterok:$id_ek_upd --array=1-$n -n $n_proc_scm $thisdir/parallel_scm_eval.sbatch $it $job_id $n_proc_scm $thisdir $use_expansion)
    fi
    # Update ensemble
    id_ek_upd=$(sbatch -o $logdir/%x-%j.out --parsable --partition=expansion --kill-on-invalid-dep=yes --dependency=afterok:$id_ens_array --export=n=$n $thisdir/step_ekp.sbatch $it $job_id $thisdir $use_expansion)
    done


fi


