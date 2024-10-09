

include("/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/postprocessing.jl")
using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl")

# run_setups = valid_experiment_setups
run_setups = [
    # ("SOCRATES_Base", "pow_icenuc_autoconv_eq"),
    # ("SOCRATES_Base", "tau_autoconv_noneq"),
    # ("SOCRATES_exponential_T_scaling_ice", "tau_autoconv_noneq"),
    # ("SOCRATES_exponential_T_scaling_ice_raw", "tau_autoconv_noneq"),
    # ("SOCRATES_powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    # ("SOCRATES_geometric_liq__geometric_ice", "tau_autoconv_noneq"),
    # ("SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice", "tau_autoconv_noneq") ,
    # ("SOCRATES_geometric_liq__powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    # ("SOCRATES_neural_network", "tau_autoconv_noneq"),
    # ("SOCRATES_linear_combination", "tau_autoconv_noneq"),
    # ("SOCRATES_linear_combination_with_w", "tau_autoconv_noneq"),
]
run_setups = run_setups_default

run_postprocessing_runs = false
if run_postprocessing_runs
    for (experiment, setup) in run_setups
        for calibration_vars in run_calibration_vars_list
            calibration_vars_str = join(sort(calibration_vars), "__")
            CEDMF_output_dir    = joinpath(experiments_dir, "subexperiments", experiment,"Calibrate_and_Run", setup, calibration_vars_str, "calibrate", "output", "Atlas_LES", "RFAll_obs")
            # CEDMF_output_data_dir    = joinpath(experiments_dir, "subexperiments", experiment,"Data_Storage", "Calibrate_and_Run", setup, calibration_vars_str, "calibrate", "output", "Atlas_LES", "RFAll_obs")

            # ----- make sure Data_Storage exists in each subexperiment postprocessing folder for storage ---------------------------------------------- #
            postprocessing_experiment_dir = joinpath(postprocess_runs_storage_dir, "subexperiments", experiment) # `run` is kind of a catchall for any runs we do, so we want our own specific dir for post processing probably...
            mkpath(postprocessing_experiment_dir)
            @info("postprocessing_experiment_dir: $postprocessing_experiment_dir")
            run(`cd $postprocessing_experiment_dir`); cd(postprocessing_experiment_dir)
            rm("$postprocessing_experiment_dir/Data_Storage", force=true) # remove symlink if it exists
            # mkpath(joinpath(postprocessing_experiment_dir, "subexperiments", experiment) )
            run(`ln -sf ../../Data_Storage/subexperiments/$experiment $postprocessing_experiment_dir/Data_Storage`)


            # -------------- now make sure the output folder exists in the postprocessing folder and the data_storage path ------------------------------------- #
            # postprocessing_experiment_dir              = joinpath(postprocess_dir, "subexperiments", experiment,                 "Calibrate_and_Run", setup, calibration_vars_str, "postprocessing") # `run` is kind of a catchall for any runs we do, so we want our own specific dir for post processing probably...
            # postprocessing_experiment_data_storage_dir = joinpath(postprocess_dir, "subexperiments", experiment, "Data_Storage", "Calibrate_and_Run", setup, calibration_vars_str, "postprocessing") # `run` is kind of a catchall for any runs we do, so we want our own specific dir for post processing probably...

            postprocessing_experiment_dir              = joinpath(postprocess_runs_storage_dir, "subexperiments", experiment,                 "Calibrate_and_Run", setup, calibration_vars_str, "postprocessing") # `run` is kind of a catchall for any runs we do, so we want our own specific dir for post processing probably...
            postprocessing_experiment_data_storage_dir = joinpath(postprocess_runs_storage_dir, "subexperiments", experiment, "Data_Storage", "Calibrate_and_Run", setup, calibration_vars_str, "postprocessing") # `run` is kind of a catchall for any runs we do, so we want our own specific dir for post processing probably...
            mkpath(postprocessing_experiment_dir)
            mkpath(joinpath(postprocessing_experiment_data_storage_dir, "output") ) # make output path w/ data_storage

            # now we gotta create a simlink to a path in data_storage
            working_dir = abspath(pwd())
            run(`cd $postprocessing_experiment_dir`); cd(postprocessing_experiment_dir)
            rm("$postprocessing_experiment_dir/output", force=true) # remove symlink if it exists
            run(`ln -sf ../../../../Data_Storage/Calibrate_and_Run/$setup/$calibration_vars_str/postprocessing/output  $postprocessing_experiment_dir/output`) # could also use  postprocessing_experiment_data_storage_dir
            run(`cd $working_dir`); cd(working_dir)
            # --------------------------------------------------- #
            save_dir = joinpath(postprocessing_experiment_dir, "output", "Atlas_LES", "RFAll_obs")

            postprocess_run( CEDMF_output_dir, save_dir, )

        end
    end
end


# figure out how to collate the results as we desire :)

#= I think for flexibility we should keep a different item for each
<experiment>/<setup>/<calibration_vars_str>/<method> just like w/ the output runs
=#
run_collation_process = false
test_out = nothing
if run_collation_process
    for (experiment, setup) in run_setups
        for calibration_vars in run_calibration_vars_list
            calibration_vars_str = join(sort(calibration_vars), "__")
            CEDMF_output_dir        = joinpath(experiments_dir, "subexperiments", experiment,"Calibrate_and_Run", setup, calibration_vars_str, "calibrate", "output", "Atlas_LES", "RFAll_obs") # the dir where the Diagnostics output is
            # postprocess_output_dir  = joinpath(postprocess_dir, "subexperiments", experiment,"Calibrate_and_Run", setup, calibration_vars_str, "postprocessing", "output", "Atlas_LES", "RFAll_obs") # the dir where the postprocessed output is
            postprocess_output_dir  = joinpath(postprocess_runs_storage_dir, "subexperiments", experiment,"Calibrate_and_Run", setup, calibration_vars_str, "postprocessing", "output", "Atlas_LES", "RFAll_obs") # the dir where the postprocessed output is, is outside SOCRATES for rsync efficiency
            save_dir                = joinpath(postprocess_dir, "subexperiments", experiment,"Calibrate_and_Run", setup, calibration_vars_str, "postprocessing", "output", "Atlas_LES", "RFAll_obs") # the dir where the collated output will be saved

            reprocess_reference = false
            delete_TC_output_files = false # testing
            # alt_postprocess_output_dir = replace(postprocess_output_dir, "SOCRATES" => "SOCRATES_postprocess_runs_storage", count=1) # replace first occurence, (so not the SOCRATES in subexperiment) | store elsewhere for rsync reasons (would need to change in runs above too)

            # global test_out = collate_postprocess_runs( CEDMF_output_dir, postprocess_output_dir, save_dir; overwrite = true, overwrite_reference = reprocess_reference , delete_TC_output_files = delete_TC_output_files  ) # this is kinda slow so maybe we could launch a slurm job for each one? idk...
            # if using the slurm version, just do a manual run to make already_processed_reference true, so those files don't get rewritten over and over 
            collate_postprocess_runs_slurm( CEDMF_output_dir, postprocess_output_dir, save_dir; overwrite = true, overwrite_reference = reprocess_reference, delete_TC_output_files = delete_TC_output_files) # this is kinda slow so maybe we could launch a slurm job for each one? idk...

        end
    end
end

@info("test_out: $test_out")









