"""Evaluates a set of SCM configurations for a single parameter vector."""

@everywhere using Pkg
@everywhere Pkg.activate("../../..")
@everywhere using ArgParse
@everywhere using CalibrateEDMF
@everywhere using CalibrateEDMF.Pipeline
@everywhere using CalibrateEDMF.TurbulenceConvectionUtils

@everywhere src_dir = dirname(pathof(CalibrateEDMF))
@everywhere using CalibrateEDMF.HelperFuncs
@everywhere include(joinpath(src_dir, "parallel.jl"))
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
    default = "train"
end
parsed_args = parse_args(ARGS, s)
outdir_path = parsed_args["job_dir"]
include(joinpath(outdir_path, "config.jl"))

config = get_config()
version = parsed_args["version"]
mode = parsed_args["mode"]
versioned_model_eval_parallel(version, outdir_path, mode, config)
