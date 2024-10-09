#!/bin/bash

#SBATCH --time=2:00:00    # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --ntasks=1         # number of tasks ( 5 cases + 1 control for julia call below)
#SBATCH --mem-per-cpu=15G # memory per CPU core (10G was what i was using, for 6 jobs parallel x6? idk...)  (reduced from 60G,... i think 10-15 should work, idk why it didnt work on the julia_parallel version... used to work when run separately... -- hopefully queues faster...
#SBATCH -J "collate_postprocess"      # job name






cedmf_output_dir=${1?Error: no CEMDF output dir given}
tc_output_dir=${2?Error: no postprocessor tc_output_dir given}
save_dir=${3?Error: no save dir given}
overwrite=${4?Error: no overwrite given}
overwrte_reference=${5?Error: no overwrite reference given}
reference_files_already_have_derived_data_vars_LES=${6?Error: no reference_files_already_have_derived_data_vars_LES given}
delete_TC_output_files=${7?Error: no delete_TC_output_files given}

collate_postprocess_run_file=/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/collate_postprocess_runs.jl
CEDMF_path=/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl


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
else
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
    # julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()' # no sysimage, no skylake-avx512, # i dont think we should do this every time...
    # julia --project -e 'using Pkg; Pkg.precompile()' # I this is redundant with instantiate 

fi




# n_proc_scm=6 # 5 cases + 1 control (is this needed maybe bc TCRunner.jl is parallelized? idk... seems to have fixed world age problem... )
echo "Running postprocessor"
julia --project=$CEDMF_path $collate_postprocess_run_file --cedmf_output_dir=$cedmf_output_dir --tc_output_dir=$tc_output_dir --save_dir=$save_dir --overwrite=$overwrite --overwrite_reference=$overwrte_reference --reference_files_already_have_derived_data_vars_LES=$reference_files_already_have_derived_data_vars_LES --delete_TC_output_files=$delete_TC_output_files
echo "done"
