"""Evaluates a set of SCM configurations for a single parameter vector."""

using ArgParse
using CalibrateEDMF
include("../ParameterSweepPipeline.jl")
using Main.ParameterSweepPipeline
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
using JLD2

# Read iteration number of ensemble to be recovered
s = ArgParseSettings()
@add_arg_table s begin
    "--version"
    help = "Calibration process number"
    arg_type = Int
    "--job_dir"
    help = "Job output directory"
    arg_type = String
    default = "output"
    "--mode"
    help = "Forward model evaluation mode: `train` or `validation`"
    arg_type = String
end
parsed_args = parse_args(ARGS, s)
outdir_path = parsed_args["job_dir"]
include(joinpath(outdir_path, "config.jl"))

config = get_config()
version = parsed_args["version"]
versioned_model_eval(version, outdir_path, config)
