using ArgParse
using CalibrateEDMF.TurbulenceConvectionUtils
include("DiagnosticsTools.jl")

s = ArgParseSettings()
@add_arg_table s begin
    "--results_dir"
    help = "Path to the results directory (CEDMF output)"
    arg_type = String
    "--output_dir"
    help = "Path to store TC output"
    arg_type = String
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
    run_TC_optimal_val(results_dir::String, output_dir::String, config::Dict; method::String = "best_nn_particle_mean", metric::String = "mse_full" )
Given path to the results directory of completed calibration run and associated config, run TC
    using optimal paramters on validation dataset and save resulting TC stats files.
Inputs: 
 - results_dir - directory containing CEDMF `Diagnostics.nc` file.
 - output_dir - directory to store TC output
 - config - config dictionary
 - `method`: method for computing optimal parameters. Use parameters of:
        "best_particle" - particle with lowest mse in training (`metric` = "mse_full") or validation (`metric` = "mse_full_val") set.
        "best_nn_particle_mean" - particle nearest to ensemble mean for the iteration with lowest mse.
 - `metric` : mse metric to find the minimum of {"mse_full", "val_mse_full"}.
"""

function run_TC_optimal_val(
    results_dir::String,
    output_dir::String,
    config::Dict;
    method::String = "best_nn_particle_mean",
    metric::String = "mse_full",
)

    namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    val_config = get(config, "validation", nothing)

    u_names, u = optimal_parameters(joinpath(results_dir, "Diagnostics.nc"); method = method, metric = metric)

    for i in 1:length(val_config["case_name"])
        run_SCM_handler(
            val_config["case_name"][i],
            output_dir;
            u = u,
            u_names = u_names,
            namelist_args = namelist_args,
            uuid = val_config["scm_suffix"][i],
            les = get_stats_path(val_config["y_dir"][i]),
        )
    end

end

run_TC_optimal_val(
    parsed_args["results_dir"],
    parsed_args["output_dir"],
    config;
    method = parsed_args["method"],
    metric = parsed_args["metric"],
)
