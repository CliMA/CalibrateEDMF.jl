"""Performs a single step of the EnsembleKalmanProcess."""

# Import modules to all processes
using ArgParse
using Distributions
using StatsBase
using LinearAlgebra
using NCDatasets
const NC = NCDatasets
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.NetCDFIO
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
using JLD2
using Base


"""
    ek_update(
        ekobj::EnsembleKalmanProcess,
        priors::ParameterDistribution,
        iteration::Int64,
        config::Dict{Any, Any},
        versions::Vector{String},
        outdir_path::String,
    )

Updates an EnsembleKalmanProcess using forward model evaluations stored
in output files defined by their `versions`, and generates the parameters
for the next ensemble for forward model evaluations. The updated EnsembleKalmanProcess
and new ModelEvaluator are both written to file.

Inputs:

 - ekobj         :: EnsembleKalmanProcess to be updated.
 - priors        :: Priors over parameters, used for unconstrained-constrained mappings.
 - iteration     :: Current iteration of the calibration process.
 - config        :: Process configuration dictionary.
 - versions      :: String versions identifying the forward model evaluations.
 - outdir_path   :: Output path directory.
"""
function ek_update(
    ekobj::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    iteration::Int64,
    config::Dict{Any, Any},
    versions::Vector{String},
    outdir_path::String,
)
    Δt = config["Δt"]
    deterministic_forward_map = config["noisy_obs"]
    # Get dimensionality
    mod_evaluator = load(scm_output_path(outdir_path, versions[1]))["model_evaluator"]
    N_par, N_ens = size(ekobj.u[1])
    @assert N_ens == length(versions)
    C = sum(ref_model -> length(get_t_start(ref_model)), mod_evaluator.ref_models)
    Δt_scaled = Δt / C # Scale artificial timestep by batch size

    u = zeros(N_ens, N_par)
    g = zeros(N_ens, pca_length(mod_evaluator.ref_stats))
    for (ens_index, version) in enumerate(versions)
        scm_outputs = load(scm_output_path(outdir_path, version))
        mod_evaluator = scm_outputs["model_evaluator"]
        u[ens_index, :] = mod_evaluator.param_cons
        g[ens_index, :] = scm_outputs["g_scm_pca"]
    end
    g = Array(g')
    # Advance EKP
    if isa(ekobj.process, Inversion)
        update_ensemble!(ekobj, g, Δt_new = Δt_scaled, deterministic_forward_map = deterministic_forward_map)
    else
        update_ensemble!(ekobj, g)
    end
    ekp = ekobj
    jldsave(ekobj_path(outdir_path, iteration + 1); ekp)

    # Get new step
    params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekobj))
    params = [c[:] for c in eachcol(params_cons_i)]
    mod_evaluators =
        [ModelEvaluator(param, get_name(priors), mod_evaluator.ref_models, mod_evaluator.ref_stats) for param in params]
    versions = map(mod_eval -> generate_scm_input(mod_eval, outdir_path), mod_evaluators)
    # Store version identifiers for this ensemble in a common file
    write_versions(versions, iteration + 1, outdir_path = outdir_path)

    # Diagnostics IO
    update_diagnostics(outdir_path, ekp, priors)
    return
end

"""
    update_diagnostics(outdir_path::String, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Appends current iteration diagnostics to a diagnostics netcdf file.

    Inputs:
    - outdir_path :: Path of results directory.
    - ekp :: Current EnsembleKalmanProcess.
    - priors:: Prior distributions of the parameters.
"""
function update_diagnostics(outdir_path::String, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    diags = NetCDFIO_Diags(joinpath(outdir_path, "Diagnostics.nc"))
    open_files(diags)
    io_metrics(diags, ekp)
    io_particle_diags(diags, ekp, priors)
    close_files(diags)
end


# Read iteration number of ensemble to be recovered
s = ArgParseSettings()
@add_arg_table s begin
    "--iteration"
    help = "Calibration iteration number"
    arg_type = Int
    "--config"
    help = "Inverse problem config file"
    arg_type = String
    "--job_dir"
    help = "Job output directory"
    arg_type = String
    default = "output"
end
parsed_args = parse_args(ARGS, s)

include(parsed_args["config"])

# Recover inputs for update
iteration = parsed_args["iteration"]
outdir_path = parsed_args["job_dir"]
versions = readlines(joinpath(outdir_path, "versions_$(iteration).txt"))
priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))
ekobj = load(ekobj_path(outdir_path, iteration))["ekp"]
process_config = get_config()["process"]
ek_update(ekobj, priors, iteration, process_config, versions, outdir_path)
