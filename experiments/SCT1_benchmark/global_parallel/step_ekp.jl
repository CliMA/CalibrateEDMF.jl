"""Performs a single step of the EnsembleKalmanProcess."""

# Import modules to all processes
using ArgParse
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.Pipeline
src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using JLD2

# Read iteration number of ensemble to be recovered
s = ArgParseSettings()
@add_arg_table s begin
    "--iteration"
    help = "Calibration iteration number"
    arg_type = Int
    "--job_dir"
    help = "Job output directory"
    arg_type = String
    default = "output"
end
parsed_args = parse_args(ARGS, s)
# Recover inputs for update
iteration = parsed_args["iteration"]
outdir_path = parsed_args["job_dir"]

include(joinpath(outdir_path, "config.jl"))

versions = readlines(joinpath(outdir_path, "versions_$(iteration).txt"))
priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))
ekobj = load(ekobj_path(outdir_path, iteration))["ekp"]
config = get_config()
ek_update(ekobj, priors, iteration, config, versions, outdir_path)
