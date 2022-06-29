"""Restarts a calibration process."""

# Import modules to all processes
using ArgParse
using Glob
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.Pipeline
using CalibrateEDMF.HelperFuncs
# Import EKP modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using JLD2

s = ArgParseSettings()
@add_arg_table s begin
    "--output_dir"
    help = "Results output directory"
    arg_type = String
    "--job_id"
    help = "Job identifier"
    arg_type = String
    default = "12345"
end
parsed_args = parse_args(ARGS, s)
# Recover inputs for restart
outdir_path = parsed_args["output_dir"]
include(joinpath(outdir_path, "config.jl"))
ekobjs = glob("ekobj_iter_*.jld2", outdir_path)
iters = @. parse(Int64, getfield(match(r"(?<=ekobj_iter_)(\d+)", basename(ekobjs)), :match))
last_iteration = maximum(iters)

priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))
ekobj = load(ekobj_path(outdir_path, last_iteration))["ekp"]
config = get_config()
restart_calibration(ekobj, priors, last_iteration, config, outdir_path; job_id = parsed_args["job_id"])
