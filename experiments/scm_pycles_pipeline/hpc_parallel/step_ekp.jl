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
    ref_stats = mod_evaluator.ref_stats
    ref_models = mod_evaluator.ref_models
    _, N_ens = size(get_u_final(ekobj))
    @assert N_ens == length(versions)
    C = length(ref_stats.pca_vec)
    Δt_scaled = Δt / C # Scale artificial timestep by batch size

    g = zeros(pca_length(ref_stats), N_ens)
    g_full = zeros(full_length(ref_stats), N_ens)
    for (ens_index, version) in enumerate(versions)
        scm_outputs = load(scm_output_path(outdir_path, version))
        g[:, ens_index] = scm_outputs["g_scm_pca"]
        g_full[:, ens_index] = scm_outputs["g_scm"]
        # Clean up
        rm(scm_output_path(outdir_path, version))
        rm(scm_init_path(outdir_path, version))
    end

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
    mod_evaluators = [ModelEvaluator(param, get_name(priors), ref_models, ref_stats) for param in params]
    versions = map(mod_eval -> generate_scm_input(mod_eval, outdir_path), mod_evaluators)
    # Store version identifiers for this ensemble in a common file
    write_versions(versions, iteration + 1, outdir_path = outdir_path)

    # Diagnostics IO
    update_diagnostics(outdir_path, ekp, priors, ref_stats, g_full)
    return
end

"""
    update_diagnostics(outdir_path::String, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Appends current iteration diagnostics to a diagnostics netcdf file.

    Inputs:
    - outdir_path :: Path of results directory.
    - ekp :: Current EnsembleKalmanProcess.
    - priors:: Prior distributions of the parameters.
    - ref_stats :: ReferenceStatistics.
    - g_full :: The forward model evaluation in primitive space.
"""
function update_diagnostics(
    outdir_path::String,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    ref_stats::ReferenceStatistics,
    g_full::Array{FT, 2},
) where {FT <: Real}

    # Compute diagnostics
    error_full = compute_mse(g_full, ref_stats.y_full)
    diags = NetCDFIO_Diags(joinpath(outdir_path, "Diagnostics.nc"))
    open_files(diags)
    io_metrics(diags, ekp, error_full)
    io_particle_diags(diags, ekp, priors, g_full, error_full)
    close_files(diags)
end


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
process_config = get_config()["process"]
ek_update(ekobj, priors, iteration, process_config, versions, outdir_path)
