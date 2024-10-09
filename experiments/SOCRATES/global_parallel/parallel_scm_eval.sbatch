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
    # module load climacommon # doesnt work bc of nsys
    # module load julia/1.10.1 
    # module load julia-preferences/2024_02_20
    # module load mpiwrapper/2024_02_27 # is this all we need for mpi? the code still complained about libmpi not being found but it still ran? idk...
    module load climacommon
fi

run_num=${SLURM_ARRAY_TASK_ID}
job_dir=$(head $job_id".txt" | tail -1)
version=$(head -"$run_num" $job_dir/"versions_"$iteration_".txt" | tail -1)

echo $job_id
echo $version
echo $PWD
echo $job_dir


# julia --project -C skylake-avx512 -JCEDMF.so -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train" && (
#   echo sysimage loaded successfully
# ) || (
#   julia --project -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train"
# ) &

julia --project -p $((n_proc_scm/2)) $thisdir/parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "train" & # ignore system image for now (according to costa)
P1=$!
# julia --project -C skylake-avx512 -JCEDMF.so -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation" && (
#   echo sysimage loaded successfully
# ) || (
#   julia --project -p $((n_proc_scm/2)) parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation"
# ) &

julia --project -p $((n_proc_scm/2)) $thisdir/parallel_scm_eval.jl --version $version --job_dir $job_dir --mode "validation" & # ignore system image for now (according to costa)
P2=$!
wait $P1 $P2
echo "SCM simulation for ${version} in iteration ${iteration_} finished"



if [ "$use_expansion" = false ]; then
    echo "w"
else
    echo "l"
fi