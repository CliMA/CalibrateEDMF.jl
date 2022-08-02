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
using CalibrateEDMF.ModelTypes
# Import EKP modules
using JLD2
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

# Cases defined as structs for quick access to default configs
struct ObsCampaigns end
struct LesDrivenScm end
struct LesDrivenScmVal end

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem
    config["reference"] = get_reference_config(ObsCampaigns())
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
    config["overwrite_scm_file"] = true # Flag for overwritting SCM input file
    return config
end

function get_regularization_config()
    config = Dict()
    # Regularization of observations: mean and covariance
    config["perform_PCA"] = true # Performs PCA on data
    config["variance_loss"] = 1.0e-2 # Variance truncation level in PCA
    config["normalize"] = true  # whether to normalize data by pooled variance
    config["tikhonov_mode"] = "relative" # Tikhonov regularization
    config["tikhonov_noise"] = 1.0e-5 # Tikhonov regularization
    config["dim_scaling"] = true # Dimensional scaling of the loss

    # Parameter regularization: L2 regularization with respect to prior mean
    # Set to `nothing` or `0.0` to use prior covariance as regularizer.
    # To turn off regularization, set config["process"]["augmented"] to false.
    config["l2_reg"] = 1.0 #0.1 #1.0

    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 45 #15 (for no batching)
    config["N_ens"] = 9 #29 #50 # Must be 2p+1 when algorithm is "Unscented"
    config["algorithm"] = "Unscented" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = 1.0
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = true
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    return config
end

function get_reference_config(::ObsCampaigns)
    config = Dict()
    config["case_name"] = ["DYCOMS_RF02", "Rico", "TRMM_LBA"]

    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] = [
        ["thetal_mean", "ql_mean", "qt_mean", "qr_mean"],
        ["thetal_mean", "ql_mean", "qt_mean", "qr_mean"],
        ["thetal_mean", "ql_mean", "qi_mean", "qt_mean", "qr_mean", "qs_mean"],
    ]
    config["y_dir"] = [
        "/groups/esm/ajaruga/KK_2000_acnv/Output.DYCOMS_RF02.20f711ab-e6f7-4cd6-b638-5184f1837550/",
        "/groups/esm/ajaruga/KK_2000_acnv/Output.Rico.2beaa716-3af9-4c13-97c8-98260ffd9eb7/",
        "/groups/esm/ajaruga/KK_2000_acnv/Output.TRMM_LBA.6a117167-ef7a-4931-bb02-c6c9886547f9/",
    ]

    config["scm_suffix"] = repeat(["000000"], length(config["case_name"]))
    config["scm_parent_dir"] = repeat(["scm_init"], length(config["case_name"]))
    config["t_start"] = [4, 20, 4] * 3600.0
    config["t_end"] = [6, 24, 6] * 3600.0

    config["Σ_t_start"] = [0, 12, 2] * 3600.0
    config["Σ_t_end"] = [6, 24, 6] * 3600.0

    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # microphysics parameters
        "A_acnv_KK2000" => [bounded(1e13, 1e12)],
        "a_acnv_KK2000" => [bounded(0.1, 10.0)],
        "b_acnv_KK2000" => [bounded(-10.0, -0.1)],
        "c_acnv_KK2000" => [bounded(-10.0, -0.1)],
    )

    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # microphysics parameters
        "A_acnv_KK2000" => [7.42e13],
        "a_acnv_KK2000" => [2.47],
        "b_acnv_KK2000" => [-1.79],
        "c_acnv_KK2000" => [-1.47],
    )

    # For Inversion should be 1, for Unscented 0.25 (tight prior)
    config["unconstrained_σ"] = 0.25
    return config
end

function get_scm_config()
    config = Dict()
    config["namelist_args"] = [
        ("time_stepping", "adapt_dt", true),
        ("time_stepping", "dt_max", 4.0),
        ("time_stepping", "dt_min", 1.0),
        ("stats_io", "frequency", 60.0),
        ("turbulence", "EDMF_PrognosticTKE", "updraft_number", 1),
        ("turbulence", "EDMF_PrognosticTKE", "Prandtl_number_0", 0.74),
        ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_adv_coeff", 0.001),
        ("turbulence", "EDMF_PrognosticTKE", "entrainment_factor", 0.13),
        ("turbulence", "EDMF_PrognosticTKE", "detrainment_factor", 0.51),
        ("turbulence", "EDMF_PrognosticTKE", "entrainment_smin_tke_coeff", 0.3),
        ("turbulence", "EDMF_PrognosticTKE", "updraft_mixing_frac", 0.25),
        ("turbulence", "EDMF_PrognosticTKE", "entrainment_scale", 0.0004),
        ("turbulence", "EDMF_PrognosticTKE", "sorting_power", 2.0),
        ("turbulence", "EDMF_PrognosticTKE", "turbulent_entrainment_factor", 0.075),
        ("turbulence", "EDMF_PrognosticTKE", "tke_ed_coeff", 0.14),
        ("turbulence", "EDMF_PrognosticTKE", "tke_diss_coeff", 0.22),
        ("turbulence", "EDMF_PrognosticTKE", "static_stab_coeff", 0.4),
        ("turbulence", "EDMF_PrognosticTKE", "tke_surf_scale", 3.75),
        ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_buoy_coeff1", 0.12),
        ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_drag_coeff", 10.0),
        ("turbulence", "EDMF_PrognosticTKE", "surface_area", 0.1),
        ("thermodynamics", "sgs", "quadrature"),
        ("thermodynamics", "quadrature_order", 3),
        ("thermodynamics", "quadrature_type", "log-normal"),
        ("microphysics", "precipitation_model", "clima_1m"),
        ("microphysics", "prescribed_Nd", 55e6 ),
        ("microphysics", "rain_formation_scheme", "KK2000"),
        ("microphysics", "precip_fraction_model", "cloud_cover"),
        ("microphysics", "precip_fraction_limiter", 0.3),
        ("microphysics", "microph_scaling", 1.0),
        ("microphysics", "microph_scaling_dep_sub", 1.0),
        ("microphysics", "microph_scaling_melt", 1.0),
        ("microphysics", "microph_scaling_acnv", 1.0),
        ("microphysics", "microph_scaling_accr", 1.0),
        ("microphysics", "E_liq_rai", 0.8),
        ("microphysics", "E_liq_sno", 0.1),
        ("microphysics", "E_ice_rai", 1.0),
        ("microphysics", "E_ice_sno", 0.1),
        ("microphysics", "E_rai_sno", 1.0),
    ]
    return config
end
