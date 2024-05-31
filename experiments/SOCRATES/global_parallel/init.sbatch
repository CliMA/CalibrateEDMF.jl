#!/bin/bash

#SBATCH --time=2:00:00      # walltime (what times do we need for our runs)
#SBATCH --ntasks=1          # number of processor cores (i.e. tasks)
#SBATCH --nodes=1           # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH -J "sct_init"       # job name



config=${1?Error: no config file given}
job_id=${2?Error: no job ID given}
thisdir=${3?Error: Working directory not given}

# check if fourth argument is given (use_expansion)
if [ -z "$4" ]; then
    use_expansion=false
else
    use_expansion=${4}
fi

# call Sys.CPU_NAME in julia and get the architecture
# architecture=$(julia -e 'using Sys; println(Sys.CPU_NAME)')
# sys_image_name = CEDMF_${architecture}.so



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

# julia --project -C skylake-avx512 -JCEDMF.so init.jl --config $config --job_id $job_id && (
#   echo sysimage loaded successfully
# ) || (
#   julia --project init.jl --config $config --job_id $job_id
# )


julia --project $thisdir/init.jl --config $config --job_id $job_id 

echo 'Ensemble initialized for calibration.'

