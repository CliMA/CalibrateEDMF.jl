"""Initializes a SCM calibration process."""

using ArgParse
using CalibrateEDMF
include("../Pipeline.jl")
using Main.ParameterSweepPipeline

s = ArgParseSettings()
@add_arg_table s begin
    "--config"
    help = "Inverse problem config file"
    arg_type = String
    "--job_id"
    help = "Job identifier"
    arg_type = String
    default = "12345"
end
parsed_args = parse_args(ARGS, s)
include(parsed_args["config"])
config = get_config()

init_sweep(config; job_id = parsed_args["job_id"], config_path = parsed_args["config"])
