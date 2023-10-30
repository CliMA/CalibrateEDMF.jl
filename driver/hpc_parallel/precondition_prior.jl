"""Evaluates the stability of a set of SCM configurations for a parameter vector and possibly draws a new one."""

using ArgParse
using Distributions
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.HelperFuncs
# Import EKP modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using JLD2


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
end
parsed_args = parse_args(ARGS, s)
version = parsed_args["version"]
outdir_path = parsed_args["job_dir"]
include(joinpath(outdir_path, "config.jl"))
config = get_config()
proc_config = config["process"]
max_counter = get_entry(proc_config, "precondition_max_counter", 10)

scm_args = load(scm_init_path(outdir_path, version))
priors = load(joinpath(outdir_path, "prior.jld2"))["prior"]
ekobj = load(ekobj_path(outdir_path, 1))["ekp"]

# Preconditioning ensemble methods in unconstrained space
if isa(ekobj.process, Inversion) || isa(ekobj.process, Sampler)
    model_evaluator = precondition(scm_args["model_evaluator"], priors; max_counter = max_counter)
    batch_indices = scm_args["batch_indices"]
    rm(scm_init_path(outdir_path, version))
    jldsave(scm_init_path(outdir_path, version); model_evaluator, version, batch_indices)
end
