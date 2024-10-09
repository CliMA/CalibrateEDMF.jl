#!/bin/bash

# Don't forget to change to 14 hours one day


new_calibration_vars_list = (
    ["ql_mean", "qi_mean"],
    ["ql_all_mean", "qi_all_mean"],
    ["temperature_mean", "ql_mean", "qi_mean"],
    ["temperature_mean", "ql_all_mean", "qi_all_mean"],
)


experiments = (
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


setups = ("pow_icenuc_autoconv_eq", "tau_autoconv_noneq")



valid_experiment_setups = (
    ("SOCRATES_Base", "tau_autoconv_noneq"),
    ("SOCRATES_Base", "pow_icenuc_autoconv_eq"),
    ("SOCRATES_exponential_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_exponential_T_scaling_ice_raw", "tau_autoconv_noneq"),
    ("SOCRATES_powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__geometric_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_neural_network", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination_with_w", "tau_autoconv_noneq"),
)


# run_setups = valid_experiment_setups
run_setups = [
    # ("SOCRATES_Base", "tau_autoconv_noneq"),
    # ("SOCRATES_Base", "pow_icenuc_autoconv_eq"),
    # ("SOCRATES_exponential_T_scaling_ice", "tau_autoconv_noneq"),
    # ("SOCRATES_exponential_T_scaling_ice_raw", "tau_autoconv_noneq"),
    ("SOCRATES_powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__geometric_ice", "tau_autoconv_noneq"),
    # ("SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice", "tau_autoconv_noneq") ,
    # ("SOCRATES_geometric_liq__exponential_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    # ("SOCRATES_neural_network", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination_with_w", "tau_autoconv_noneq"),
]

# run_setups = [
#     ("SOCRATES_neural_network", "tau_autoconv_noneq"),
# ]


# run_setups = [
#     # ("SOCRATES_Base", "tau_autoconv_noneq"),
#     ("SOCRATES_Base", "pow_icenuc_autoconv_eq"),
#     # ("SOCRATES_exponential_T_scaling_ice", "tau_autoconv_noneq"),
#     # ("SOCRATES_exponential_T_scaling_ice_raw", "tau_autoconv_noneq"),
#     # ("SOCRATES_powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
#     # ("SOCRATES_geometric_liq__geometric_ice", "tau_autoconv_noneq"),
#     # ("SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice", "tau_autoconv_noneq") ,
#     # ("SOCRATES_geometric_liq__powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
#     # ("SOCRATES_neural_network", "tau_autoconv_noneq"),
#     # ("SOCRATES_linear_combination", "tau_autoconv_noneq"),
#     # ("SOCRATES_linear_combination_with_w", "tau_autoconv_noneq"),
# ]


# dt_min_list = [2.0,]
dt_min_list = [5.0]
# dt_min_list = [0.5, 2.0, 5.0] # factor of 10 increase is ~factor 10 decrease in number of vertical points.... we'll have to see how that looks.....
dt_max_list = dt_min_list .* 4.0
adapt_dt_list = repeat([true], length(dt_min_list))

dt_setups = zip(dt_min_list, dt_max_list, adapt_dt_list)

# run_calibration_vars_list = new_calibration_vars_list
run_calibration_vars_list = [
    # ["ql_mean", "qi_mean"],
    # ["ql_all_mean", "qi_all_mean"],
    ["ql_mean", "qi_mean", "qr_mean", "qip_mean"],
    # ["temperature_mean", "ql_mean", "qi_mean"],
    # ["temperature_mean", "ql_all_mean", "qi_all_mean"],
]

# This runs into QOS limits if you submit too many jobs... so you may need to limit jobs somehow...

default_resource = Dict(
    "time" => 7200, # seconds (time for precompilation)
    "memory" => 15 * 1024, # MB (base memory for running any such job in julia, we know 2 is too low...
    "precondition_memory" => 60 * 1024, # MB (base memory for running any such job in julia, we know 2 is too low...
)
min_resource = Dict(
    "time" => 2400, # seconds (20 mins for precomp, 20 mins for running) - we know 20 mins total is too little for some runs...
    "memory" => 6 * 1024, # MB (base memory for running any such job in julia, we know 2GB per core is too low bc we saw crashes..., we know we use about 6GB per core @ dt=5
    "precondition_memory" => 6 * 1024, # MB (base memory for running any such job in julia, we know 2GB per core is too low bc we saw crashes..., we know we use about 6GB per core @ dt=5
)


let
    local num_my_init_processes::Int
    local in_the_pipeline_jobs_list::Vector{Int} = []
    local in_the_pipeline_jobs::Int
    local qos_process_limit::Int = 10000 # quality of service job limit

    local N_ens::Int
    local use_ens_param_factor::Bool
    local N_iter::Int

    local resource_scaling_factors = Dict("time" => Dict(), "memory" => Dict(), "precondition_memory" => Dict())


    local min_resource_scaling_factor = Dict(
        "time" => min_resource["time"] / default_resource["time"],
        "memory" => min_resource["memory"] / default_resource["memory"],
        "precondition_memory" => min_resource["precondition_memory"] / default_resource["precondition_memory"],
    )


    #= Notes
    dt_min=0.5
    -

    dt_min=0.2
    - job runs used about 3.5 GB per core (how can this be less than the dt=5 runs?????)
    - job runs took about 20 mins

    dt_min=5.0
    - precond used ~7GB, max of around 30 mins
    - runs used about 6GB per core, took about 15 mins though some took 30+
    - run used over 3.5 GB per cpu...

    =#

    local max_preconditioner_counter = 2 # can move up again maybe since we reduced our priors a bit but really does drag out the calibrations...
    local SLURM_RESTART_COUNT_limit = 0 # Haven't seen as many random failures lately... allow 1 restart...


    for (experiment, setup) in run_setups

        for (dt_min, dt_max, adapt_dt) in dt_setups
            dt_string =
                adapt_dt ? "adapt_dt__dt_min_" * string(dt_min) * "__dt_max_" * string(dt_max) : "dt_" * string(dt_min)

            resource_scaling_factor_time = get(resource_scaling_factors["time"], dt_min, (0.5 / dt_min)^1) # one power for timestepping, one power for vertical resolution... However, for dt_min = 2, seff suggested 23% memory utilization before this scaling and jobs taking about 20 minutes at most... default was 2 hours before... so that's a reduction of about 6 times, not 16... (some is probably precomp overhead need to account for that...)
            resource_scaling_factor_mem = get(resource_scaling_factors["memory"], dt_min, (0.5 / dt_min)^1) # one power for timestepping, one power for vertical resolution... However, for dt_min = 2, seff suggested 23% memory utilization before this scaling and jobs taking about 20 minutes at most... default was 2 hours before... so that's a reduction of about 6 times, not 16... (some is probably precomp overhead need to account for that...)
            resource_scaling_factor_precondition_mem =
                get(resource_scaling_factors["precondition_memory"], dt_min, (0.5 / dt_min)^1) # one power for timestepping, one power for vertical resolution... However, for dt_min = 2, seff suggested 23% memory utilization before this scaling and jobs taking about 20 minutes at most... default was 2 hours before... so that's a reduction of about 6 times, not 16... (some is probably precomp overhead need to account for that...)

            @info(
                "Pre-adjustment resource_scaling_factor_time: $resource_scaling_factor_time, Pre-adjustment resource_scaling_factor_mem: $resource_scaling_factor_mem",
                "Pre-adjustment resource_scaling_factor_precondition_mem: $resource_scaling_factor_precondition_mem"
            )


            # if resource_scaling_factor_time < min_resource_scaling_factor["time"]
            #     @warn("resource_scaling_factor_time: $resource_scaling_factor_time is less than min_resource_scaling_factor_time: $(min_resource_scaling_factor["time"]), adjusting to yield at least $(min_resource["time"]) seconds per job") 
            #     resource_scaling_factor_time = min_resource_scaling_factor["time"]
            # end
            # if resource_scaling_factor_mem < min_resource_scaling_factor["memory"]
            #     @warn("resource_scaling_factor_mem: $resource_scaling_factor_mem is less than min_resource_scaling_factor_mem: $(min_resource_scaling_factor["memory"]), adjusting to yield at least $(min_resource["memory"]) MB per cpu") 
            #     resource_scaling_factor_mem = min_resource_scaling_factor["memory"]
            # end

            # scale the the part above the minimum
            new_time =
                min_resource["time"] + (default_resource["time"] - min_resource["time"]) * resource_scaling_factor_time
            resource_scaling_factor_time = new_time / default_resource["time"] # is always > min_resource_scaling_factor["time"] by default

            new_mem =
                min_resource["memory"] +
                (default_resource["memory"] - min_resource["memory"]) * resource_scaling_factor_mem
            resource_scaling_factor_mem = new_mem / default_resource["memory"] # is always > min_resource_scaling_factor["memory"] by default

            new_precondition_mem =
                min_resource["precondition_memory"] +
                (default_resource["precondition_memory"] - min_resource["precondition_memory"]) *
                resource_scaling_factor_precondition_mem
            resource_scaling_factor_precondition_mem = new_precondition_mem / default_resource["precondition_memory"] # is always > min_resource_scaling_factor["memory"] by default

            @info(
                "Adjusted resource_scaling_factor_time: $resource_scaling_factor_time, Adjusted resource_scaling_factor_mem: $resource_scaling_factor_mem, Adjusted resource_scaling_factor_precondition_mem: $resource_scaling_factor_precondition_mem"
            )

            for calibration_vars in run_calibration_vars_list
                calibration_vars_str = join(sort(calibration_vars), "__")

                num_my_processes =
                    parse(Int, readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r`, `wc -l`)))
                num_my_launch_and_call_processes =
                    parse(
                        Int,
                        readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r -n sct_call`, `wc -l`)),
                    ) + parse(
                        Int,
                        readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r -n launch_calibrate`, `wc -l`)),
                    )

                # @info("Before: num_my_launch_and_call_processes, $num_my_launch_and_call_processes, in_the_pipeline_jobs_list: $in_the_pipeline_jobs_list")
                # how many init processes running that may not have launched their jobs yet
                if length(in_the_pipeline_jobs_list) > 0
                    in_the_pipeline_jobs_list =
                        in_the_pipeline_jobs_list[(end - num_my_launch_and_call_processes + 1):end] # remove the jobs that have already been launched 
                end
                in_the_pipeline_jobs = sum(in_the_pipeline_jobs_list)

                # @info("num_my_processes: $num_my_processes,  num_my_launch_and_call_processes, $num_my_launch_and_call_processes, in_the_pipeline_jobs: $in_the_pipeline_jobs, in_the_pipeline_jobs_list: $in_the_pipeline_jobs_list")

                config_file = joinpath(
                    "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments",
                    experiment,
                    "Calibrate_and_Run",
                    setup,
                    dt_string,
                    calibration_vars_str,
                    "calibrate",
                    "configs",
                    "config_calibrate_RFAll_obs.jl",
                )
                @info("config_file: $config_file")

                for line in eachline(config_file)
                    if occursin(r"N_ens\s+=", line)
                        N_ens = parse(Int, filter(isdigit, split(line, "=")[2])) # should really be a split at comment character '#' like above  in case comment includes digits, like      parse(Bool,  split(line, ('=', '#'))[2])
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
                    @info(
                        "Waiting to launch jobs for experiment ($experiment, $setup, $dt_string, $calibration_vars_str) beacause adding requested $N_to_launch processes to the current running/pending $num_my_processes and soon to be launched $in_the_pipeline_jobs processes would exceed the Slurm QOS process limit of $qos_process_limit"
                    )
                    sleep(60)
                    num_my_processes =
                        parse(Int, readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r`, `wc -l`)))
                    num_my_launch_and_call_processes =
                        parse(
                            Int,
                            readchomp(pipeline(`squeue -u jbenjami -h -t pending,running -r -n sct_call`, `wc -l`)),
                        ) + parse(
                            Int,
                            readchomp(
                                pipeline(`squeue -u jbenjami -h -t pending,running -r -n launch_calibrate`, `wc -l`),
                            ),
                        ) # how many of our runs haven't launched their jobs yet
                    if length(in_the_pipeline_jobs_list) > 0
                        in_the_pipeline_jobs_list =
                            in_the_pipeline_jobs_list[(end - num_my_launch_and_call_processes + 1):end] # remove the jobs that have already been launched 
                    end
                    in_the_pipeline_jobs = sum(in_the_pipeline_jobs_list)
                    flush(stdout) # force writing to stdout now instead of buffering
                    flush(stderr) # force writing to stderr now instead of buffering (including logging)
                end

                @info("Launching $N_to_launch jobs for ($experiment, $setup, $dt_string, $calibration_vars_str)")
                run(
                    `sbatch -p expansion /home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh $experiment $setup $dt_string $calibration_vars_str $max_preconditioner_counter $SLURM_RESTART_COUNT_limit $resource_scaling_factor_time $resource_scaling_factor_mem $resource_scaling_factor_precondition_mem `,
                ) # tilde didn't work
                push!(in_the_pipeline_jobs_list, N_to_launch) # add to the list of jobs in the pipeline
                sleep(0.5) # give sbatch a second to launch the jobs
                flush(stdout) # force writing to stdout now instead of buffering
                flush(stderr) # force writing to stderr now instead of buffering (including logging)
            end
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
