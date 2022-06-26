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
using CalibrateEDMF.HelperFuncs
# Import EKP modules
using JLD2
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

struct SCT1Train end
struct SCT1Val end

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(SCT1Train())
    # Define reference used for validation
    config["validation"] = get_reference_config(SCT1Val())
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
    config["outdir_root"] = pwd()
    return config
end

function get_regularization_config()
    config = Dict()
    # Regularization of observations: mean and covariance
    config["perform_PCA"] = true # Performs PCA on data
    config["variance_loss"] = 1.0e-2 # Variance truncation level in PCA
    config["normalize"] = true  # whether to normalize data by pooled variance
    config["tikhonov_mode"] = "relative" # Tikhonov regularization
    config["tikhonov_noise"] = 1.0e-6 # Tikhonov regularization
    config["dim_scaling"] = true # Dimensional scaling of the loss

    # Parameter regularization: L2 regularization with respect to prior mean
    # Set to `nothing` or `0.0` to use prior covariance as regularizer.
    # To turn off regularization, set config["process"]["augmented"] to false.
    config["l2_reg"] = 0.1
    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 50
    config["N_ens"] = 140 # Must be 2p+1 when algorithm is "Unscented"
    config["algorithm"] = "Inversion" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = 1.0
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = true
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    return config
end

function get_reference_config(::SCT1Train)
    config = Dict()

    # AMIP data: October
    cfsite_numbers = (17, 18, 20, 22, 23)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 10, experiment = "amip")
    ref_dirs = [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers]

    n_repeat = length(ref_dirs)
    config["case_name"] = repeat(["LES_driven_SCM"], n_repeat)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] =
        repeat([["s_mean", "ql_mean", "qt_mean", "total_flux_qt", "total_flux_s", "u_mean", "v_mean"]], n_repeat)
    config["y_dir"] = ref_dirs
    config["t_start"] = repeat([3.0 * 3600], n_repeat)
    config["t_end"] = repeat([6.0 * 3600], n_repeat)
    # Use full LES timeseries for covariance
    config["Σ_t_start"] = repeat([-5.75 * 24 * 3600], n_repeat)
    config["Σ_t_end"] = repeat([6.0 * 3600], n_repeat)
    config["write_full_stats"] = false
    return config
end

function get_reference_config(::SCT1Val)
    config = Dict()

    # AMIP4K data: July
    cfsite_numbers = (3, 5, 8, 11, 14)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 7, experiment = "amip4K")
    ref_dirs = [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers]

    n_repeat = length(ref_dirs)
    config["case_name"] = repeat(["LES_driven_SCM"], n_repeat)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] =
        repeat([["s_mean", "ql_mean", "qt_mean", "total_flux_qt", "total_flux_s", "u_mean", "v_mean"]], n_repeat)
    config["y_dir"] = ref_dirs
    config["t_start"] = repeat([3.0 * 3600], n_repeat)
    config["t_end"] = repeat([6.0 * 3600], n_repeat)
    # Use full LES timeseries for covariance
    config["Σ_t_start"] = repeat([-5.75 * 24 * 3600], n_repeat)
    config["Σ_t_end"] = repeat([6.0 * 3600], n_repeat)
    config["write_full_stats"] = false
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # entrainment parameters
        "nn_ent_params" => [repeat([no_constraint()], 69)...],
        "turbulent_entrainment_factor" => [bounded(0.0, 10.0)],
    )

    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # entrainment parameters
        "nn_ent_params" => 0.1 .* (rand(69) .- 0.5),
        "turbulent_entrainment_factor" => [0.075],
    )

    config["unconstrained_σ"] = 1.0
    return config
end

function get_scm_config()
    config = Dict()
    config["namelist_args"] = [
        ("time_stepping", "dt_min", 1.0),
        ("time_stepping", "dt_max", 2.0),
        ("stats_io", "frequency", 60.0),
        ("grid", "nz", 80),
        ("turbulence", "EDMF_PrognosticTKE", "entrainment", "NN"),
        ("turbulence", "EDMF_PrognosticTKE", "area_limiter_power", 0.0),
        ("turbulence", "EDMF_PrognosticTKE", "entr_dim_scale", "inv_z"),
        ("turbulence", "EDMF_PrognosticTKE", "nn_ent_biases", true),
    ]
    return config
end
