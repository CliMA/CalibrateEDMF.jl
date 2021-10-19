"""Initializes a SCM calibration process."""

using ArgParse
using CalibrateEDMF
using CalibrateEDMF.Pipeline

# Include calibration config file to define problem
include("config.jl")

s = ArgParseSettings()
@add_arg_table s begin
    "--n_ens"
    help = "Number of ensemble members."
    arg_type = Int
    "--n_it"
    help = "Number of algorithm iterations."
    arg_type = Int
    "--job_id"
    help = "Job identifier"
    arg_type = String
    default = "default_id"
end
parsed_args = parse_args(ARGS, s)

init_calibration(parsed_args["n_ens"], parsed_args["n_it"], parsed_args["job_id"], get_config())
