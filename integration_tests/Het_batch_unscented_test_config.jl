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
struct ObsCampaigns end
struct ObsCampaignsVal end
# struct MyAwesomeSetup end

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(ObsCampaigns())
    # Define reference used for validation
    config["validation"] = get_reference_config(ObsCampaignsVal())
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
    config["tikhonov_noise"] = 1.0e-6 # Tikhonov regularization
    config["dim_scaling"] = true # Dimensional scaling of the loss

    # Parameter regularization: L2 regularization with respect to prior mean.
    #  - Set to `nothing` to use prior covariance as regularizer,
    #  - Set to a float for isotropic parameter regularization.
    #  - Pass a dictionary of lists similar to config["prior_mean"] for
    #       anisotropic regularization. The dictionary must be complete.
    #       If you want to avoid regularizing a certain parameter, set the entry
    #       to [0].
    # To turn off regularization, set config["process"]["augmented"] to false.
    config["l2_reg"] = 1.0
    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 6
    config["N_ens"] = 7
    config["algorithm"] = "Unscented" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = 1.0
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = true
    config["failure_handler"] = "sample_succ_gauss"
    return config
end

function get_reference_config(::ObsCampaigns)
    config = Dict()
    config["case_name"] = ["GABLS", "DYCOMS_RF01"]
    # Flag to indicate source of data (LES or SCM) for reference data and covariance
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] = [["thetal_mean", "u_mean", "v_mean", "tke_mean"], ["thetal_mean", "ql_mean", "qt_mean"]]
    config["y_dir"] = ["/groups/esm/ilopezgo/Output.GABLS.iles128wCov", "/groups/esm/ilopezgo/Output.DYCOMS_RF01.may20"]
    # provide list of dirs if different from `y_dir`
    # config["Σ_dir"] = [...]
    config["scm_suffix"] = repeat(["000123"], length(config["case_name"]))
    config["scm_parent_dir"] = repeat(["scm_init"], length(config["case_name"]))
    config["t_start"] = [7, 2] * 3600.0
    config["t_end"] = [9, 4] * 3600.0
    # Specify averaging intervals for covariance, if different from mean vector (`t_start` & `t_end`)
    # config["Σ_t_start"] = [...]
    # config["Σ_t_end"] = [...]
    config["batch_size"] = 1
    return config
end

function get_reference_config(::ObsCampaignsVal)
    config = Dict()
    config["case_name"] = ["DYCOMS_RF01"]
    # Flag to indicate source of data (LES or SCM) for reference data and covariance
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] = [["total_flux_h", "total_flux_qt"]]
    config["y_dir"] = ["/groups/esm/ilopezgo/Output.DYCOMS_RF01.may20"]
    # provide list of dirs if different from `y_dir`
    # config["Σ_dir"] = [...]
    config["scm_suffix"] = repeat(["000123"], length(config["case_name"]))
    config["scm_parent_dir"] = repeat(["scm_init"], length(config["case_name"]))
    config["t_start"] = [2] * 3600.0
    config["t_end"] = [4] * 3600.0
    # Specify averaging intervals for covariance, if different from mean vector (`t_start` & `t_end`)
    # config["Σ_t_start"] = [...]
    # config["Σ_t_end"] = [...]
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # diffusion parameters
        "tke_ed_coeff" => [bounded(0.01, 1.0)],
        "tke_diss_coeff" => [bounded(0.01, 1.0)],
        "static_stab_coeff" => [bounded(0.1, 1.0)],
    )
    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # diffusion parameters
        "tke_ed_coeff" => [0.05],
        "tke_diss_coeff" => [0.5],
        "static_stab_coeff" => [0.9],
    )

    config["unconstrained_σ"] = 0.5
    return config
end

function get_scm_config()
    config = Dict()
    config["namelist_args"] =
        [("time_stepping", "dt_min", 1.0), ("time_stepping", "dt_max", 10.0), ("stats_io", "frequency", 60.0)]
    return config
end
