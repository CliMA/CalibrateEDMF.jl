"""
We want to take our output runs and postprocess them to get the statistics we want.

Run simulations for:
    1. The last iteration ensemble
    2. The best iteration ensemble member (if it's not in the last iteration, otherwise just copy it over)
    3. The ensemble mean parameters of each iteration (or just the last one? idk...) -- we have the ensemble mean of all the individual ensemble runs already... 

For all these, copy over the loss/mse from Diagnostics.nc 
    - We wont have this for ensemble means in Diagnostic.nc, but we can calculate it the same way we did for the plots from the 
    - For the runs w/ ensemble mean paramters, we would need to invert norm_factors etc to get to unscaled space and calculate these values


We can store the full runs somewhere, but then we will want to take means over the SOCRATES_summary reference time period (do we want variances too?)
While doing this, we can calculate anything additional we want to save
    - e.g. derived microphyiscal process rates (from model and truth)
    - e.g. sedimentation rates (from model and from truth)
    - e.g. sublimation/deposition rates (from model and from truth)
    - e.g. combined liquid and ice category values

    We will also want to copy over the loss/mse from Diagnostics.nc (or can we use the norm_factors to go to constrained space and recalculate it?)

What do we do about failures lol?

"""



#=
We need a simple script that takes an in path for the diagnostics.nc and an outpath for where to save and runs all the simulations we need and saves them w/ correct names etc...
    - but we need many runs w/ different nameslists, etc... how to do?
    - can we use the TC runner to do this? i thought those all ran on the same node? 
    - we need many runs w/ different parameter values -- just look up how it's being done rn and copy that...
       - we do want to make sure we save them w/ identifiable names for reference in postprocessing.jl

    - they use a slurm job array and then pull the arguments based on the job number... -- but how can we store the arguments we need? they are just using the same config and then letting random sampling do the rest... (not sure where they define the new dists but)
    - maybe it's easiest still then to just loop over and submit each one individually...? How to store the namelist then? or can we write the namelists to namelist_##.in files and then use job array?

    - the namelist isn't stored w/ diagnostics.nc though... so we need to generate it from config
=#

CEDMF_dir = "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl"
experiments_dir = joinpath(CEDMF_dir, "experiments/SOCRATES")
# postprocess_dir = joinpath(experiments_dir, "Postprocessing") # eh we're gonna move this to be elsewhere... also we need data_storage
postprocess_dir = joinpath(CEDMF_dir, "experiments/SOCRATES")
# Underneath here keep the same structure as the experiments directory, subexperiments etc...



# run postprocessing runs
for (experiment, setup) in run_setups
    for calibration_vars in run_calibration_vars_list
        calibration_vars_str = join(sort(calibration_vars), "__")

        ensemble_size = # read from somewhere

        # we will have list of these so that we can use a slurm job array -- you'll take your index i and read the jld file

        methods = [
            "best_particle",
            "best_nn_particle_mean",
            "best_particle_final",
            "mean_best_ensemble",
            "mean_final_ensemble",
            ("best_ensemble_mean_$(i)" for i in 1:ensemble_size)...,
            ]

        

        out_paths = [joinpath(postprocess_dir, "subexperiments", experiment, setup, calibration_vars_str, method) for method in methods]
        map(out_path -> mkdir(out_path, force=true), out_paths)
        mkdir(joinpath(postprocess_dir, "subexperiments", experiment, setup, calibration_vars_str, "best_ensemble_mean")) # this directory wont have a run in it per se, but mean of the best_ensemble_mean_$(i) runs would go here...

        namelist_paths = [joinpath(out_path, "namelist.in") for out_path in out_paths]


        namelist_paths_file = joinpath(postprocess_dir, "subexperiments", experiment, setup, calibration_vars_str, "namelist_paths.jld2")
        out_paths_file = joinpath(postprocess_dir, "subexperiments", experiment, setup, calibration_vars_str, "out_paths.jld2")

        # ============================================================================ #
        #= NOTE: Honestly, the otherway is probably better bc we can use optimal_parameters(), otherwise I need to figure out how to get optimal_parameters into a namelist file for this one =#
        #= Their way also automatically pulls the config file etc to construct namelist and everythihng =#
        #= It's also set up to run on the entire validation set which would be annoying to set up yourself? =#
        # ============================================================================ #

        # == create namelists ============================================================ #
        namelists = [] # list of namelists for each run (create from config somehow)
        # == update namelists ============================================================ #
        optimal_parameters_lists = []
        for (i_m, method) in enumerate(methods)
            # path_to_diagnostics_file = some_path
            optimal_parameters(path_to_diagnostics_file, method = method, metric = "mse_full")


            namelist = namelists[i_m]
            # copied from run_SCM_handler() in TurbulenceConvectionUtils.jl
            u_names, u = create_parameter_vectors(u_names, u, param_map, namelist)
            # update learnable parameter values
            @assert length(u_names) == length(u)
            for (pName, pVal) in zip(u_names, u)
                param_subdict = namelist_subdict_by_key(namelist, pName)
                param_subdict[pName] = pVal
            end
        end

        # ============================================================================ #

        # just need script that takes namelist and runs the simulation and saves it to the outpath

        # then another that does the time averaging we need...
        # ============================================================================ #

        
        # run the postprocessing runs w/ TC runner (can we collect them all up into a list of sbatch commands and then submit them?)
        # run(`sbatch -p expansion /a/b/script_path.sh $diagnostics_path $out_dir`) #

        run(`id_launch_runs=\$(sbatch -o $logdir/%x-%A_%a.out --parsable --partition=expansion --kill-on-invalid-dep=yes --array=1-$n /a/b/script_path.sh $namelist_paths_file $out_paths_file)`) # this script needs to take its run_num=${SLURM_ARRAY_TASK_ID} and read namelist_paths_file and out_paths_file to get the correct paths to the namelist and out files
        run(`id_postprocess_runs=\$(sbatch -o $logdir/%x-%A_%a.out --parsable --partition=expansion --kill-on-invalid-dep=yes --dependency=afterok:$id_launch_runs --array=1-$n /a/b/script_path.sh $namelist_paths_file $out_paths_file)`) # this script needs to take its run_num=${SLURM_ARRAY_TASK_ID} and read namelist_paths_file and out_paths_file to get the correct paths to the namelist and out files

    end
end

