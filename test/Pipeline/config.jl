#= Custom calibration configuration file. =#

using Distributions
using StatsBase
using LinearAlgebra
using CalibrateEDMF
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.LESUtils
using CalibrateEDMF.TurbulenceConvectionUtils
const src_dir = dirname(pathof(CalibrateEDMF))
using CalibrateEDMF.HelperFuncs
# Import EKP modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using JLD2

namelist_args = [
    ("time_stepping", "t_max", 2.0 * 3600),
    ("time_stepping", "dt_max", 30.0),
    ("time_stepping", "dt_min", 20.0),
    ("stats_io", "frequency", 120.0),
    ("grid", "dz", 150.0),
    ("grid", "nz", 20),
]

# Cases defined as structs for quick access to default configs
struct Bomex end

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(Bomex())
    # Define the parameter priors
    config["prior"] = get_prior_config()
    # Define the kalman process
    config["process"] = get_process_config()
    # Define the SCM static configuration
    config["scm"] = get_scm_config()
    return config
end

function get_output_config()
    config = Dict()
    config["outdir_root"] = mktempdir()
    return config
end

function get_regularization_config()
    config = Dict()
    config["perform_PCA"] = true # Performs PCA on data
    config["variance_loss"] = 1.0e-2 # Variance truncation level in PCA
    config["normalize"] = true  # whether to normalize data by pooled variance
    config["tikhonov_mode"] = "relative" # Tikhonov regularization
    config["tikhonov_noise"] = 1.0e-4 # Tikhonov regularization
    config["dim_scaling"] = false # Dimensional scaling of the loss
    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 3
    config["N_ens"] = 5
    config["algorithm"] = "Inversion" # "Sampler", "Unscented"
    return config
end

function get_reference_config(::Bomex)
    config = Dict()
    config["case_name"] = ["Bomex"]
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = SCM()
    config["Σ_reference_type"] = SCM()
    config["y_names"] = [["thetal_mean", "qt_mean"]]
    ref_root_dir = mktempdir()
    config["y_dir"] = [joinpath(ref_root_dir, "Output.Bomex.ref")]
    config["t_start"] = [0.0]
    config["t_end"] = [2.0 * 3600]
    config["namelist_args"] = [namelist_args]
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] =
        Dict("entrainment_factor" => [bounded(0.0, 0.5)], "detrainment_factor" => [bounded(0.0, 0.5)])
    config["unconstrained_σ"] = 0.5
    return config
end

function get_scm_config()
    config = Dict()
    config["namelist_args"] = nothing
    return config
end
