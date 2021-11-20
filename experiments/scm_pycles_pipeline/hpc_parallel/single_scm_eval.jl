"""Evaluates a set of SCM configurations for a single parameter vector."""

using ArgParse
using Distributions
using CalibrateEDMF
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
include(joinpath(outdir_path, "config.jl"))

scm_args = load(scm_init_path(outdir_path, version))
namelist_args = get_config()["scm"]["namelist_args"]
model_evaluator = scm_args["model_evaluator"]
sim_dirs, g_scm, g_scm_pca = run_SCM(model_evaluator, namelist_args = namelist_args)

jldsave(scm_output_path(outdir_path, version); sim_dirs, g_scm, g_scm_pca, model_evaluator, version)
