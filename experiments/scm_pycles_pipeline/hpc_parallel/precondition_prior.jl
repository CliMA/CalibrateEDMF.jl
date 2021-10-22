"""Evaluates the stability of a set of SCM configurations for a parameter vector and possibly draws a new one."""

using ArgParse
using Distributions
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
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

scm_args = load(scm_init_path(outdir_path, version))
priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))
u_names = scm_args["u_names"]
ref_models = scm_args["ref_models"]
ref_stats = scm_args["ref_stats"]

# Preconditioning in unconstrained space
u_unconstrained = precondition(
    transform_constrained_to_unconstrained(priors, scm_args["u"]),
    priors,
    map(x -> deserialize_struct(x, ReferenceModel), ref_models),
    deserialize_struct(ref_stats, ReferenceStatistics),
)
# Transform back to constrained space for SCM runs
u = transform_unconstrained_to_constrained(priors, u_unconstrained)
rm(scm_init_path(outdir_path, version))
jldsave(scm_init_path(outdir_path, version); u, u_names, ref_models, ref_stats, version)
