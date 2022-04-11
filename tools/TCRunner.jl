using ArgParse
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.HelperFuncs
include("DiagnosticsTools.jl")

s = ArgParseSettings()
@add_arg_table s begin
    "--results_dir"
    help = "Path to the results directory (CEDMF output)"
    arg_type = String
    "--tc_output_dir"
    help = "Path to store TC output"
    arg_type = String
    "--run_set"
    help = "Set of cases to run TC. {reference, validation, test}."
    arg_type = String
    default = "validation"
    "--run_set_config"
    help = "Path to config if test is used for --run_set."
    arg_type = String
    default = nothing
    "--method"
    help = "Method for computing optimal parameters"
    arg_type = String
    default = "best_nn_particle_mean"
    "--metric"
    help = "mse metric to find the minimum of"
    arg_type = String
    default = "mse_full"
end
parsed_args = parse_args(ARGS, s)
include(joinpath(parsed_args["results_dir"], "config.jl"))
config = get_config()

"""
    run_TC_optimal(results_dir::String, tc_output_dir::String, config::Dict; method::String = "best_nn_particle_mean", metric::String = "mse_full" )
Given path to the results directory of completed calibration run, run TC on a given set of cases (`run_cases`).
    using optimal parameters and save resulting TC stats files.
Inputs: 
 - results_dir - directory containing CEDMF `Diagnostics.nc` file.
 - tc_output_dir - directory to store TC output
 - namelist_args - additional arguments passed to the TurbulenceConvection namelist.
 - run_cases - Dictionary of cases to run. Must contain {"case_name","scm_suffix", "y_dir"}.
 - `method`: method for computing optimal parameters. Use parameters of:
        "best_particle" - particle with lowest mse in training (`metric` = "mse_full") or validation (`metric` = "val_mse_full") set.
        "best_nn_particle_mean" - particle nearest to ensemble mean for the iteration with lowest mse.
 - `metric` : mse metric to find the minimum of {"mse_full", "val_mse_full"}.
"""

function run_TC_optimal(
    results_dir::String,
    tc_output_dir::String,
    namelist_args::Union{Vector, Nothing} = nothing,
    run_cases::Union{Dict, Nothing} = nothing;
    method::String = "best_nn_particle_mean",
    metric::String = "mse_full",
)

    u_names, u = optimal_parameters(joinpath(results_dir, "Diagnostics.nc"); method = method, metric = metric)

    for i in 1:length(run_cases["case_name"])
        @info "Running TC suffix: $(run_cases["scm_suffix"][i])"
        les_path = if run_cases["case_name"][i] == "LES_driven_SCM"
            get_stats_path(run_cases["y_dir"][i])
        else
            nothing
        end
        run_SCM_handler(
            run_cases["case_name"][i],
            tc_output_dir;
            u = u,
            u_names = u_names,
            namelist_args = namelist_args,
            uuid = run_cases["scm_suffix"][i],
            les = les_path,
        )
    end
end


if parsed_args["run_set"] in ("reference", "validation")
    namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    run_cases = get(config, parsed_args["run_set"], nothing)
    @info "Running TC on $(parsed_args["run_set"]) set"
    run_TC_optimal(
        parsed_args["results_dir"],
        parsed_args["tc_output_dir"],
        namelist_args,
        run_cases;
        method = parsed_args["method"],
        metric = parsed_args["metric"],
    )

elseif parsed_args["run_set"] == "test"

    if isnothing(parsed_args["run_set_config"])
        error("--run_set_config must be specified if --run_set is test")
    else
        include(parsed_args["run_set_config"])
        run_cases = get_reference_config(ScmTest())
    end
    namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    @info "Running TC on $(parsed_args["run_set"]) set"
    run_TC_optimal(
        parsed_args["results_dir"],
        parsed_args["tc_output_dir"],
        namelist_args,
        run_cases;
        method = parsed_args["method"],
        metric = parsed_args["metric"],
    )

else
    error("--run_set must be one of {reference, validation, test}")
end
