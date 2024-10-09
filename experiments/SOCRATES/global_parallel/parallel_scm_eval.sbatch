#!/bin/bash

#SBATCH --time=2:00:00    # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=15G # memory per CPU core (10G was what i was using, for 6 jobs parallel x6? idk...)  (reduced from 60G,... i think 10-15 should work, idk why it didnt work on the julia_parallel version... used to work when run separately... -- hopefully queues faster...
#SBATCH -J "sct_run"      # job name

# module load julia/1.9.3 openmpi/4.1.5
# module load julia/1.10.0 mpich/4.0.0 # copied from other experiments


iteration_=${1?Error: no iteration given}
job_id=${2?Error: no job ID given}
n_proc_scm=${3?Error: number of processes not given}
thisdir=${4?Error: Working directory not given}


# check if fifth argument is given (use_expansion)
if [ -z "$5" ]; then
    use_expansion=false
else
    use_expansion=${5}
fi



if [ -z "$6" ]; then
    SLURM_RESTART_COUNT_limit=2 # Won't this cause repeated running w/ same failing parameters? or does failing on parameters count as a clean exit? (using this bc of random failures writing to disk, maybe bc of NFS? etc...)
else
    SLURM_RESTART_COUNT_limit=${6}
fi




if $(grep -q "Red Hat" /etc/redhat-release);  then # copied from /groups/esm/slurm-buildkite/hooks/environment
    os_type="rhel9" # "We are on a RedHat 9 node"
else 
	os_type="centos7" # some of the expansion nodes still seem to be running centos7 for some strange reason? idk...
fi

if [ "$use_expansion" = false ] || [ "$os_type" = "centos7" ]; then # use_expansion is false or os_type is centos7 despite being on expansion node (idk why that is happening lol)
    #julia package management
    module purge
    module load julia/1.10.0 mpich/4.0.0 # copied from other experiments
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
fi

run_num=${SLURM_ARRAY_TASK_ID}
job_dir=$(head $job_id".txt" | tail -1)
version=$(head -"$run_num" $job_dir/"versions_"$iteration_".txt" | tail -1)

echo $job_id
echo $version
echo $PWD
echo $job_dir

if [ -z "$SLURM_RESTART_COUNT" ]; then
    SLURM_RESTART_COUNT=0
fi
if [ $SLURM_RESTART_COUNT -le $SLURM_RESTART_COUNT_limit ]; then
    echo "SLURM_RESTART_COUNT is $SLURM_RESTART_COUNT, less than or equal to the limit of $SLURM_RESTART_COUNT_limit, trying..."

    # julia --project -C skylake-avx512 -JCEDMF.so -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train" && (
    #   echo sysimage loaded successfully
    # ) || (
    #   julia --project -p $((n_proc_scm)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train"
    # ) &

    julia --project -p $((n_proc_scm)) $thisdir/parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train" & # ignore system image for now (according to costa)
    julia_exit_code_1=$?
    P1=$!
    # julia --project -C skylake-avx512 -JCEDMF.so -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation" && (
    #   echo sysimage loaded successfully
    # ) || (
    #   julia --project -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation"
    # ) &

    # Turned off validation bc were not using it and gave all n_proc_scm cores to training...
    # julia --project -p $((n_proc_scm)) $thisdir/parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation" & # ignore system image for now (according to costa)
    # julia_exit_code_2=$?
    # P2=$!
    # wait $P1 $P2

    wait $P1
    # echo "SCM simulation for ${version} in iteration ${iteration_} finished"

    # if [ $julia_exit_code_1 -eq 0 ] && [ $julia_exit_code_2 -eq 0 ]; then
    if [ $julia_exit_code_1 -eq 0 ]; then
        echo "SCM simulation for ${version} in iteration ${iteration_} finished."
    else
        if [ $SLURM_RESTART_COUNT -eq $SLURM_RESTART_COUNT_limit ]; then
            echo "Cannot retry beacuse SLURM_RESTART_COUNT is $SLURM_RESTART_COUNT, equaling the limit of $SLURM_RESTART_COUNT_limit, exiting..."
        else
            echo "SCM simulation for ${version} in iteration ${iteration_} failed with exit code $julia_exit_code_1, requeuing job $SLURM_JOB_ID"
            scontrol requeue $SLURM_JOB_ID # combat random failures
        fi
        # echo "SCM simulation for ${version} in iteration ${iteration_} failed with exit codes $julia_exit_code_1 and $julia_exit_code_2, requeuing job $SLURM_JOB_ID"
        # echo "SCM simulation for ${version} in iteration ${iteration_} failed with exit code $julia_exit_code_1 requeuing job $SLURM_JOB_ID"
        # scontrol requeue $SLURM_JOB_ID # combat random failures
    fi

else 
    # echo "SLURM_RESTART_COUNT is $SLURM_RESTART_COUNT, equaling the limit of $SLURM_RESTART_COUNT_limit, exiting..."
    echo "Error, this should not happen, SLURM_RESTART_COUNT is $SLURM_RESTART_COUNT, equaling the limit of $SLURM_RESTART_COUNT_limit, exiting..."
fi


printf "\n...Done!\n\n"
