# This is an example on training the TurbulenceConvection.jl implementation
# of the EDMF scheme with data generated using PyCLES (an LES solver) or
# TurbulenceConvection.jl (perfect model setting).
#
# This example parallelizes forward model evaluations using Julia's pmap() function.
# This parallelization strategy is efficient for small ensembles (N_ens < 20). For
# calibration processes involving a larger number of ensemble members, parallelization
# using HPC resources directly is advantageous.

# Import modules to all processes
@everywhere using Pkg
@everywhere Pkg.activate("../../..")
@everywhere using ArgParse
@everywhere using CalibrateEDMF
@everywhere using CalibrateEDMF.DistributionUtils
@everywhere using CalibrateEDMF.Pipeline
@everywhere const src_dir = dirname(pathof(CalibrateEDMF))
@everywhere include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
@everywhere using EnsembleKalmanProcesses
@everywhere using EnsembleKalmanProcesses.ParameterDistributions
# include(joinpath(@__DIR__, "../../../src/viz/ekp_plots.jl"))
using JLD2

s = ArgParseSettings()
@add_arg_table s begin
    "--config"
    help = "Inverse problem config file path"
    arg_type = String
end
parsed_args = parse_args(ARGS, s)
include(parsed_args["config"])
config = get_config()

# Initialize calibration process
outdir_path = init_sweep(config; config_path = parsed_args["config"], mode = "pmap")
priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))

# Dispatch SCM eval functions to workers
@everywhere scm_eval_train(version) = versioned_model_eval(version, $outdir_path, "train", $config)
@everywhere scm_eval_validation(version) = versioned_model_eval(version, $outdir_path, "validation", $config)

# Calibration process
N_iter = config["process"]["N_iter"]
for iteration in 1:N_iter
    versions = readlines(joinpath(outdir_path, "versions_$(iteration).txt"))
    ekp = load(ekobj_path(outdir_path, iteration))["ekp"]
    pmap(scm_eval_train, versions)
    pmap(scm_eval_validation, versions)
    ek_update(ekp, priors, iteration, config, versions, outdir_path)
end
