#= Custom calibration configuration file. =#

using Distributions
using StatsBase
using LinearAlgebra
using CalibrateEDMF
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.LESUtils
using CalibrateEDMF.TurbulenceConvectionUtils
# Import prior constraints
using EnsembleKalmanProcesses.ParameterDistributions
using JLD2

# Cases defined as structs for quick access to default configs
struct Bomex end
struct ValidateBomex end

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(Bomex())
    # Define reference models to use for validation
    config["validation"] = get_reference_config(ValidateBomex())
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
    config["overwrite_scm_file"] = false # Flag for overwritting SCM input file
    return config
end

function get_regularization_config()
    config = Dict()
    # Regularization of observations: mean and covariance
    config["perform_PCA"] = true # Performs PCA on data
    config["variance_loss"] = 1.0e-2 # Variance truncation level in PCA
    config["normalize"] = true  # whether to normalize data by pooled variance
    config["tikhonov_mode"] = "relative" # Tikhonov regularization
    config["tikhonov_noise"] = 1.0e-4 # Tikhonov regularization
    config["dim_scaling"] = true # Dimensional scaling of the loss

    # Parameter regularization: L2 regularization with respect to prior mean.
    #  - Set to `nothing` to use prior covariance as regularizer,
    #  - Set to a float for isotropic parameter regularization.
    #  - Pass a dictionary of lists similar to config["prior_mean"] for
    #       anisotropic regularization. The dictionary must be complete.
    #       If you want to avoid regularizing a certain parameter, set the entry
    #       to [0].
    # To turn off regularization, set config["process"]["augmented"] to false.
    config["l2_reg"] = nothing
    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 10
    config["N_ens"] = 10
    config["algorithm"] = "Inversion" # "Sampler", "Unscented"
    config["noisy_obs"] = false
    # Artificial time stepper of the EKI.
    config["Δt"] = 1.0
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = false
    config["failure_handler"] = "sample_succ_gauss"
    return config
end

function get_reference_config(::Bomex)
    config = Dict()
    config["case_name"] = ["Bomex"]
    # Flag to indicate source of data (LES or SCM) for reference data and covariance
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    # Fields to learn from during training
    config["y_names"] = [["thetal_mean", "ql_mean", "qt_mean"]]
    # LES data can be stored as an Artifact and downloaded lazily
    config["y_dir"] = [LESUtils.get_path_to_artifact()]
    # provide list of dirs if different from `y_dir`
    # config["Σ_dir"] = [...]
    config["scm_suffix"] = ["000000"]
    config["scm_parent_dir"] = ["scm_init"]
    config["t_start"] = [4.0 * 3600]
    config["t_end"] = [6.0 * 3600]
    # Specify averaging intervals for covariance, if different from mean vector (`t_start` & `t_end`)
    # config["Σ_t_start"] = [...]
    # config["Σ_t_end"] = [...]
    # If isnothing(config["batch_size"]), use all data per iteration
    config["batch_size"] = nothing
    return config
end

function get_reference_config(::ValidateBomex)
    config = Dict()
    config["case_name"] = ["Bomex"]
    # Flag to indicate source of data (LES or SCM) for reference data and covariance
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    # Validate on different variables for this example
    config["y_names"] = [["total_flux_h", "total_flux_qt", "u_mean", "v_mean"]]
    config["y_dir"] = [LESUtils.get_path_to_artifact()]
    # provide list of dirs if different from `y_dir`
    # config["Σ_dir"] = [...]
    config["scm_suffix"] = ["000000"]
    config["scm_parent_dir"] = ["scm_init"]
    config["t_start"] = [4.0 * 3600]
    config["t_end"] = [6.0 * 3600]
    # Specify averaging intervals for covariance, if different from mean vector (`t_start` & `t_end`)
    # config["Σ_t_start"] = [...]
    # config["Σ_t_end"] = [...]
    # If isnothing(config["batch_size"]), use all data per iteration
    config["batch_size"] = nothing
    return config
end

function get_prior_config()
    config = Dict()
    # Define prior bounds on the parameters.
    config["constraints"] =
        Dict("entrainment_factor" => [bounded(0.0, 0.5)], "detrainment_factor" => [bounded(0.3, 0.8)])
    # Define prior mean (must be within bounds).
    config["prior_mean"] = Dict("entrainment_factor" => [0.02], "detrainment_factor" => [0.4])
    # Define width of the probability distribution with respect to the bounds. This is equivalent
    # to the σ of a Gaussian in unconstrained space.
    config["unconstrained_σ"] = 0.5
    return config
end

function get_scm_config()
    config = Dict()
    # List of tuples like [("time_stepping", "dt_min", 1.0)], or nothing
    config["namelist_args"] = [("stats_io", "calibrate_io", true)]
    return config
end
