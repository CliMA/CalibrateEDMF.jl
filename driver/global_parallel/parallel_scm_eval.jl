"""Evaluates a set of SCM configurations for a single parameter vector."""

# Read iteration number of ensemble to be recovered
s = ArgParseSettings()
@add_arg_table s begin
    "--version"
    help = "Calibration process number"
    arg_type = String
    "--job_dir"
    help = "Job output directory"
    arg_type = String
    default = "output"
    "--mode"
    help = "Forward model evaluation mode: `train` or `validation`"
    arg_type = String
    default = "train"
end
parsed_args = parse_args(ARGS, s)
outdir_path = parsed_args["job_dir"]
include(joinpath(outdir_path, "config.jl"))

config = get_config()
version = parsed_args["version"]
mode = parsed_args["mode"]
versioned_model_eval_parallel(version, outdir_path, mode, config)
