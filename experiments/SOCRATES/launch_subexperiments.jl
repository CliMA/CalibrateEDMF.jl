#!/bin/bash

# Don't forget to change to 14 hours one day


new_calibration_vars_list = (
    ["ql_mean", "qi_mean"],
    ["ql_all_mean", "qi_all_mean"],
    ["temperature_mean", "ql_mean", "qi_mean"],
    ["temperature_mean", "ql_all_mean", "qi_all_mean"],
)


experiments=(
    "SOCRATES_Base",
    "SOCRATES_exponential_T_scaling_ice",
    "SOCRATES_exponential_T_scaling_ice_raw",
    "SOCRATES_powerlaw_T_scaling_ice",
    "SOCRATES_geometric_liq__geometric_ice",
    "SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice",
    "SOCRATES_geometric_liq__powerlaw_T_scaling_ice",
    "SOCRATES_neural_network",
    "SOCRATES_linear_combination",
    "SOCRATES_linear_combination_with_w",
)


setups = ("pow_icenuc_autoconv_eq", "tau_autoconv_noneq",)



valid_experiment_setups = (
    ("SOCRATES_Base", "tau_autoconv_noneq"),
    ("SOCRATES_Base", "pow_icenuc_autoconv_eq"),
    ("SOCRATES_exponential_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_exponential_T_scaling_ice_raw", "tau_autoconv_noneq"),
    ("SOCRATES_powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__geometric_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice", "tau_autoconv_noneq") ,
    ("SOCRATES_geometric_liq__powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_neural_network", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination_with_w", "tau_autoconv_noneq"),
)


# run_setups = valid_experiment_setups
run_setups = [
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

# run_calibration_vars_list = new_calibration_vars_list
run_calibration_vars_list = [
    # ["ql_mean", "qi_mean"],
    # ["ql_all_mean", "qi_all_mean"],
    ["ql_mean", "qi_mean", "qr_mean", "qip_mean"],
    # ["temperature_mean", "ql_mean", "qi_mean"],
    # ["temperature_mean", "ql_all_mean", "qi_all_mean"],
]

# This runs into QOS limits if you submit too many jobs... so you may need to limit jobs somehow...

let 
    local num_my_init_processes::Int 
    local in_the_pipeline_jobs_list::Vector{Int} = []
    local in_the_pipeline_jobs::Int
    local qos_process_limit::Int = 10000 # quality of service job limit

    local N_ens::Int
    local N_iter::Int

    for (experiment, setup) in run_setups
        for calibration_vars in run_calibration_vars_list
            calibration_vars_str = join(sort(calibration_vars), "__")

            num_my_processes = parse(Int, readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r`, `wc -l`)))
            num_my_launch_and_call_processes = parse(Int, readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r -n sct_call`, `wc -l`))) + parse(Int, readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r -n launch_calibrate`, `wc -l`)))
            
            # @info("Before: num_my_launch_and_call_processes, $num_my_launch_and_call_processes, in_the_pipeline_jobs_list: $in_the_pipeline_jobs_list")
            # how many init processes running that may not have launched their jobs yet
            if length(in_the_pipeline_jobs_list) > 0
                in_the_pipeline_jobs_list = in_the_pipeline_jobs_list[(end-num_my_launch_and_call_processes+1):end] # remove the jobs that have already been launched 
            end
            in_the_pipeline_jobs = sum(in_the_pipeline_jobs_list)

            # @info("num_my_processes: $num_my_processes,  num_my_launch_and_call_processes, $num_my_launch_and_call_processes, in_the_pipeline_jobs: $in_the_pipeline_jobs, in_the_pipeline_jobs_list: $in_the_pipeline_jobs_list")

            config_file = joinpath("/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments", experiment, "Calibrate_and_Run", setup, calibration_vars_str, "calibrate", "configs", "config_calibrate_RFAll_obs.jl")
            # @info("config_file: $config_file")
            for line in eachline(config_file)
                # @info("line: $line", typeof(line))
                if occursin(r"N_ens\s+=", line)
                    N_ens = parse(Int, filter(isdigit, split(line, "=")[2]))
                    # @info("N_ens: $N_ens")
                end
                if occursin(r"N_iter\s+=", line)
                    N_iter = parse(Int, filter(isdigit, split(line, "=")[2]))
                    @info("N_iter: $N_iter")
                end          
            end

            # @info("N_ens: $N_ens, N_iter: $N_iter")

            N_to_launch = (
                1 # calling script (ekp_par_calibration.sbatch)
                + 1 # init_script (init.sbatch)
                + N_ens # preconditioner (precondion_prior.sbatch)
                + (N_ens * N_iter)  # iterations (parallel_scn_eval.sbatch)
                + N_iter # control scripts (step_ekp.sbatch)
                )

            # @info("N_to_launch: $N_to_launch")

            while (num_my_processes + N_to_launch + in_the_pipeline_jobs) > qos_process_limit
                @info("Waiting to launch jobs for experiment ($experiment, $setup, $calibration_vars_str) beacause adding requested $N_to_launch processes to the current running/pending $num_my_processes and soon to be launched $in_the_pipeline_jobs processes would exceed the Slurm QOS process limit of $qos_process_limit")
                sleep(60)
                num_my_processes = parse(Int, readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r`, `wc -l`)))
                num_my_launch_and_call_processes = parse(Int, readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r -n sct_call`, `wc -l`))) + + parse(Int, readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r -n launch_calibrate`, `wc -l`))) # how many of our runs haven't launched their jobs yet
                if length(in_the_pipeline_jobs_list) > 0
                    in_the_pipeline_jobs_list = in_the_pipeline_jobs_list[(end-num_my_launch_and_call_processes+1):end] # remove the jobs that have already been launched 
                end
                in_the_pipeline_jobs = sum(in_the_pipeline_jobs_list)
                flush(stdout) # force writing to stdout now instead of buffering
                flush(stderr) # force writing to stderr now instead of buffering (including logging)
            end

            @info("Launching $N_to_launch jobs for ($experiment, $setup, $calibration_vars_str)")
            run(`sbatch -p expansion /home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh $experiment $setup $calibration_vars_str 5`) # tilde didn't work
            push!(in_the_pipeline_jobs_list, N_to_launch) # add to the list of jobs in the pipeline
            sleep(.5) # give sbatch a second to launch the jobs
            flush(stdout) # force writing to stdout now instead of buffering
            flush(stderr) # force writing to stderr now instead of buffering (including logging)
        end
    end
end

# scancel -u jbenjami -n precond; scancel -u jbenjami -n sct_call; scancel -u jbenjami -n sct_run; scancel -u jbenjami -n sct_cont;  scancel -u jbenjami -n sct_init; scancel -u jbenjami -n sct_call; scancel -u jbenjami -n launch_subexperiments

# ========================================================================================================================= #

# restart (edit to use)
# clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/restart_global_calibrate.sh SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice tau_autoconv_noneq 5

# ========================================================================================================================= #


# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_Base
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_exponential_T_scaling_ice
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_geometric_liq__geometric_ice
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_powerlaw_T_scaling_ice

# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_geometric_liq__exponential_T_scaling
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_linear_combination
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_linear_combination_with_w
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_neural_network

# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES


