"""Evaluates the stability of a set of SCM configurations for a parameter vector and possibly draws a new one."""

using ArgParse
using Distributions
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
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
end
parsed_args = parse_args(ARGS, s)
version = parsed_args["version"]
outdir_path = parsed_args["job_dir"]
include(joinpath(outdir_path, "config.jl"))
config = get_config()
namelist_args = get_entry(config["scm"], "namelist_args", nothing)
particle_failure_fixer = get_entry(config["process"], "particle_failure_fixer", "high_loss")

scm_args = load(scm_init_path(outdir_path, version))
priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))
ekobj = load(ekobj_path(outdir_path, 1))["ekp"]

# Preconditioning ensemble methods in unconstrained space
if isa(ekobj.process, Inversion) || isa(ekobj.process, Sampler)
    model_evaluator = precondition(
        scm_args["model_evaluator"], priors, namelist_args = namelist_args,
        particle_failure_fixer = particle_failure_fixer,
    )
    rm(scm_init_path(outdir_path, version))
    jldsave(scm_init_path(outdir_path, version); model_evaluator, version)
end
