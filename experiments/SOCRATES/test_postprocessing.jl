

include("/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/postprocessing.jl")
using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl")

# ========================================================================================================================= #
# ========================================================================================================================= #
run_setups_default = [
    ("SOCRATES_Base", "tau_autoconv_noneq"),
    ("SOCRATES_Base", "pow_icenuc_autoconv_eq"),
    ("SOCRATES_exponential_T_scaling_ice", "tau_autoconv_noneq"),
    # ("SOCRATES_exponential_T_scaling_ice_raw", "tau_autoconv_noneq"),
    ("SOCRATES_powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__geometric_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice", "tau_autoconv_noneq") ,
    ("SOCRATES_geometric_liq__powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    # ("SOCRATES_neural_network", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination_with_w", "tau_autoconv_noneq"),
]
#
dt_min_list_default = [0.5, 2.0, 5.0] # factor of 10 increase is ~factor 10 decrease in number of vertical points.... we'll have to see how that looks.....
dt_max_list_default = dt_min_list_default .* 4.0
adapt_dt_list_default = repeat([true], length(dt_min_list_default))
dt_setups_default = zip(dt_min_list_default, dt_max_list_default, adapt_dt_list_default)
#
run_calibration_vars_list_default = [
    ["ql_mean", "qi_mean"],
    ["ql_all_mean", "qi_all_mean"],
    ["ql_mean", "qi_mean", "qr_mean", "qip_mean"],
    # ["temperature_mean", "ql_mean", "qi_mean"],
    # ["temperature_mean", "ql_all_mean", "qi_all_mean"],
]

# ========================================================================================================================= #
# ========================================================================================================================= #


run_setups = [
    ("SOCRATES_Base", "pow_icenuc_autoconv_eq"),
    ("SOCRATES_Base", "tau_autoconv_noneq"),
    ("SOCRATES_exponential_T_scaling_ice", "tau_autoconv_noneq"),
    # ("SOCRATES_exponential_T_scaling_ice_raw", "tau_autoconv_noneq"),
    ("SOCRATES_powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__geometric_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice", "tau_autoconv_noneq") ,
    ("SOCRATES_geometric_liq__powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_neural_network", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination_with_w", "tau_autoconv_noneq"),
]

# run_setups = valid_experiment_setups
# run_setups = run_setups_default

run_calibration_vars_list = [
    # ["ql_mean", "qi_mean"],
    # ["ql_all_mean", "qi_all_mean"],
    ["ql_mean", "qi_mean", "qr_mean", "qip_mean"],
    # ["temperature_mean", "ql_mean", "qi_mean"],
    # ["temperature_mean", "ql_all_mean", "qi_all_mean"],
]
# run_calibration_vars_list = new_calibration_vars_list
run_calibration_vars_list = run_calibration_vars_list_default



# dt_min_list = [2.0, 5.0] # the new ones
# dt_min_list = dt_min_list_default
# dt_max_list = dt_min_list .* 4.0
# adapt_dt_list = repeat([true], length(dt_min_list))
# dt_setups = zip(dt_min_list, dt_max_list, adapt_dt_list)
dt_setups = dt_setups_default

# ============================================================== # # the only one we ran for NN
run_setups = [ # the only one we ran for NN
    ("SOCRATES_neural_network", "tau_autoconv_noneq"),
]
dt_min_list = [5.0,] # the only one we ran for NN
dt_max_list = dt_min_list .* 4.0
adapt_dt_list = repeat([true], length(dt_min_list))
dt_setups = zip(dt_min_list, dt_max_list, adapt_dt_list)
run_calibration_vars_list = [
    ["ql_mean", "qi_mean", "qr_mean", "qip_mean"],
]
# ============================================================== #


run_postprocessing_runs = false
if run_postprocessing_runs
    for (experiment, setup) in run_setups
        for (dt_min, dt_max, adapt_dt) in dt_setups
            dt_string = adapt_dt ? "adapt_dt__dt_min_"*string(dt_min)*"__dt_max_"*string(dt_max) : "dt_"*string(dt_min)

            for calibration_vars in run_calibration_vars_list
                calibration_vars_str = join(sort(calibration_vars), "__")
                CEDMF_output_dir    = joinpath(experiments_dir, "subexperiments", experiment,"Calibrate_and_Run", setup, dt_string, calibration_vars_str, "calibrate", "output", "Atlas_LES", "RFAll_obs")
                # CEDMF_output_data_dir    = joinpath(experiments_dir, "subexperiments", experiment,"Data_Storage", "Calibrate_and_Run", setup, dt_string, calibration_vars_str, "calibrate", "output", "Atlas_LES", "RFAll_obs")

                # ----- make sure Data_Storage exists in each subexperiment postprocessing folder for storage ---------------------------------------------- #
                postprocessing_experiment_dir = joinpath(postprocess_runs_storage_dir, "subexperiments", experiment) # `run` is kind of a catchall for any runs we do, so we want our own specific dir for post processing probably...
                mkpath(postprocessing_experiment_dir)
                @info("postprocessing_experiment_dir: $postprocessing_experiment_dir")
                run(`cd $postprocessing_experiment_dir`); cd(postprocessing_experiment_dir)
                rm("$postprocessing_experiment_dir/Data_Storage", force=true) # remove symlink if it exists
                # mkpath(joinpath(postprocessing_experiment_dir, "subexperiments", experiment) )
                run(`ln -sf ../../Data_Storage/subexperiments/$experiment $postprocessing_experiment_dir/Data_Storage`)


                # -------------- now make sure the output folder exists in the postprocessing folder and the data_storage path ------------------------------------- #
                # postprocessing_experiment_dir              = joinpath(postprocess_dir, "subexperiments", experiment,                 "Calibrate_and_Run", setup, dt_string, calibration_vars_str, "postprocessing") # `run` is kind of a catchall for any runs we do, so we want our own specific dir for post processing probably...
                # postprocessing_experiment_data_storage_dir = joinpath(postprocess_dir, "subexperiments", experiment, "Data_Storage", "Calibrate_and_Run", setup, dt_string, calibration_vars_str, "postprocessing") # `run` is kind of a catchall for any runs we do, so we want our own specific dir for post processing probably...

                # make sure the postprocessing dir exists for this experiment
                postprocessing_experiment_dir              = joinpath(postprocess_runs_storage_dir, "subexperiments", experiment,                 "Calibrate_and_Run", setup, dt_string, calibration_vars_str, "postprocessing") # `run` is kind of a catchall for any runs we do, so we want our own specific dir for post processing probably...
                mkpath(postprocessing_experiment_dir)
                # make sure the data storag dir linked to exists
                postprocessing_experiment_data_storage_dir = joinpath(postprocess_runs_storage_data_dir, "subexperiments", experiment, "Calibrate_and_Run", setup, dt_string, calibration_vars_str)
                mkpath(postprocessing_experiment_data_storage_dir) #
                # make sure the subdir in data storage we're using exists
                postprocessing_experiment_data_storage_dir_linked = joinpath(postprocess_runs_storage_dir, "subexperiments", experiment, "Data_Storage", "Calibrate_and_Run", setup, dt_string, calibration_vars_str, "postprocessing") # `run` is kind of a catchall for any runs we do, so we want our own specific dir for post processing probably...
                mkpath(joinpath(postprocessing_experiment_data_storage_dir_linked, "output") ) # make output path w/ data_storage

                # now we gotta create a simlink to a path in data_storage
                working_dir = abspath(pwd())
                run(`cd $postprocessing_experiment_dir`); cd(postprocessing_experiment_dir)
                rm("$postprocessing_experiment_dir/output", force=true) # remove symlink if it exists
                run(`ln -sf ../../../../../Data_Storage/Calibrate_and_Run/$setup/$dt_string/$calibration_vars_str/postprocessing/output  $postprocessing_experiment_dir/output`) # could also use  postprocessing_experiment_data_storage_dir_linked
                run(`cd $working_dir`); cd(working_dir)
                # --------------------------------------------------- #
                save_dir = joinpath(postprocessing_experiment_dir, "output", "Atlas_LES", "RFAll_obs")

                postprocess_run( CEDMF_output_dir, save_dir, )

            end
        end
    end
end


# figure out how to collate the results as we desire :)

#= I think for flexibility we should keep a different item for each
<experiment>/<setup>/<dt_string>/<calibration_vars_str>/<method> just like w/ the output runs
=#
run_collation_process = true
test_out = nothing
if run_collation_process
    sleep(run_postprocessing_runs ? 60 * 3600 : 0) # sleep for 60 minutes if we just ran the postprocessing runs
    for (experiment, setup) in run_setups

        for (dt_min, dt_max, adapt_dt) in dt_setups
            dt_string = adapt_dt ? "adapt_dt__dt_min_"*string(dt_min)*"__dt_max_"*string(dt_max) : "dt_"*string(dt_min)

            for calibration_vars in run_calibration_vars_list
                calibration_vars_str = join(sort(calibration_vars), "__")
                CEDMF_output_dir        = joinpath(experiments_dir, "subexperiments", experiment,"Calibrate_and_Run", setup, dt_string, calibration_vars_str, "calibrate", "output", "Atlas_LES", "RFAll_obs") # the dir where the Diagnostics output is
                # postprocess_output_dir  = joinpath(postprocess_dir, "subexperiments", experiment,"Calibrate_and_Run", setup, dt_string, calibration_vars_str, "postprocessing", "output", "Atlas_LES", "RFAll_obs") # the dir where the postprocessed output is
                postprocess_output_dir  = joinpath(postprocess_runs_storage_dir, "subexperiments", experiment,"Calibrate_and_Run", setup, dt_string, calibration_vars_str, "postprocessing", "output", "Atlas_LES", "RFAll_obs") # the dir where the postprocessed output is, is outside SOCRATES for rsync efficiency
                save_dir                = joinpath(postprocess_dir, "subexperiments", experiment,"Calibrate_and_Run", setup, dt_string, calibration_vars_str, "postprocessing", "output", "Atlas_LES", "RFAll_obs") # the dir where the collated output will be saved

                reprocess_reference = false
                delete_TC_output_files = false # testing
                # alt_postprocess_output_dir = replace(postprocess_output_dir, "SOCRATES" => "SOCRATES_postprocess_runs_storage", count=1) # replace first occurence, (so not the SOCRATES in subexperiment) | store elsewhere for rsync reasons (would need to change in runs above too)

                # global test_out = collate_postprocess_runs( CEDMF_output_dir, postprocess_output_dir, save_dir; overwrite = true, overwrite_reference = reprocess_reference , delete_TC_output_files = delete_TC_output_files  ) # this is kinda slow so maybe we could launch a slurm job for each one? idk...
                # if using the slurm version, just do a manual run to make already_processed_reference true, so those files don't get rewritten over and over 
                collate_postprocess_runs_slurm( CEDMF_output_dir, postprocess_output_dir, save_dir; overwrite = true, overwrite_reference = reprocess_reference, delete_TC_output_files = delete_TC_output_files) # this is kinda slow so maybe we could launch a slurm job for each one? idk...

            end
        end
    end
end

@info("test_out: $test_out")









