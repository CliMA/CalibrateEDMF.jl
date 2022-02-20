using ArgParse
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
include("../ParameterSweepPipeline.jl")
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
    "--job_dir"
    help = "Job output directory"
    arg_type = String
    default = "output"
end
parsed_args = parse_args(ARGS, s)
# Recover inputs for update
outdir_path = parsed_args["job_dir"]

include(joinpath(outdir_path, "config.jl"))

versions = readlines(joinpath(outdir_path, "versions_$(iteration).txt"))
priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))
write_sweep_diagnostics(priors, versions, outdir_path)