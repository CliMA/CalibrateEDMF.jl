"""Custom calibration configuration file."""

using Distributions
using StatsBase
using LinearAlgebra
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.LESUtils
using CalibrateEDMF.TurbulenceConvectionUtils
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
using JLD2

# Cases defined as structs for quick access to default configs
struct BomexExpSDE end

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define preconditioning and regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(LesDrivenScm())
    # Define the parameter priors
    config["prior"] = get_prior_config()
    # Define the kalman process
    config["process"] = get_process_config()
    return config
end

function get_output_config()
    config = Dict()
    config["outdir_root"] = pwd()
    config["save_eki_data"] = true  # eki output
    config["save_ensemble_data"] = false  # .nc-files from each ensemble run
    config["overwrite_scm_file"] = false # Flag for overwritting SCM input file
    return config
end

function get_regularization_config()
    config = Dict()
    config["perform_PCA"] = true # Performs PCA on data
    config["normalize"] = true  # whether to normalize data by pooled variance
    config["tikhonov_noise"] = 1.0e-2 # Tikhonov regularization
    return config
end

function get_process_config()
    config = Dict()
    config["algorithm"] = Inversion() # Sampler(vcat(get_mean(priors)...), get_cov(priors))
    config["Δt"] = 1.0 # Artificial time stepper of the EKI.
    return config
end

function get_reference_config(::BomexExpSDE)
    config = Dict()
    config["case_name"] = ["Bomex"]
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["reference_type"] = :les
    config["y_names"] = [["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]]
    config["les_dir"] = ["/groups/esm/zhaoyi/pycles_clima/Output.Bomex.aug09"]
    config["scm_suffix"] = ["sde_exp_transform"]
    config["scm_parent_dir"] = ["/groups/esm/hervik/calibration"]
    config["t_start"] = [4.0 * 3600]
    config["t_end"] = [6.0 * 3600]
    return config
end


function get_prior_config()
    config = Dict()
    config["constraints"] =
        Dict(
            "entrainment_factor" => [bounded(0.0, 0.5)],
            "detrainment_factor" => [bounded(0.0, 0.5)],
        )
    config["unconstrained_σ"] = 0.5
    return config
end
