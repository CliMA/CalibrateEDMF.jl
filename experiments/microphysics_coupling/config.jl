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
    config["tikhonov_noise"] = 1.0e-6 # Tikhonov regularization
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
    config["N_ens"] = 57 #43 #33 # Must be 2p+1 when algorithm is "Unscented"
    config["algorithm"] = "Inversion" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = 1.0
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = false #true
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    return config
end


function get_reference_config(::ObsCampaigns)
    config = Dict()
    config["case_name"] = ["DYCOMS_RF02", "Rico", "TRMM_LBA"]
    #config["case_name"] = ["TRMM_LBA",]

    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] = [
        #["qr_mean",],
        ["thetal_mean", "ql_mean", "qt_mean", "qr_mean", "total_flux_h", "total_flux_qt"],
        ["thetal_mean", "ql_mean", "qt_mean", "qr_mean", "total_flux_h", "total_flux_qt"],
        ["thetal_mean", "ql_mean", "qi_mean", "qt_mean", "qr_mean", "qs_mean", "total_flux_h", "total_flux_qt"],
    ]
    config["y_dir"] = [
        "/groups/esm/ajaruga/calibration_data/DYCOMS_RF02/",
        "/groups/esm/ajaruga/calibration_data/Rico/",
        "/groups/esm/ajaruga/calibration_data/TRMM_LBA/",
    ]

    config["scm_suffix"] = repeat(["000000"], length(config["case_name"]))
    config["scm_parent_dir"] = repeat(["scm_init"], length(config["case_name"]))
    config["t_start"] = [4, 20, 4] * 3600.0
    config["t_end"] = [6, 24, 6] * 3600.0
    #config["t_start"] = [4,] * 3600.0
    #config["t_end"] = [6,] * 3600.0

    config["Σ_t_start"] = [0, 0, 0] * 3600.0
    config["Σ_t_end"] = [6, 24, 6] * 3600.0
    #config["Σ_t_start"] = [0,] * 3600.0
    #config["Σ_t_end"] = [6,] * 3600.0

    config["batch_size"] = 1

    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # entrainment parameters
        "entrainment_factor" => [bounded(0.0, 1.0)],
        "detrainment_factor" => [bounded(0.0, 1.0)],
        "entrainment_smin_tke_coeff" => [bounded(0.0, 10.0)],
        "updraft_mixing_frac" => [bounded(0.0, 1.0)],
        "entrainment_scale" => [bounded(1.0e-6, 1.0e-2)],
        "sorting_power" => [bounded(0.0, 4.0)],
        "turbulent_entrainment_factor" =>[bounded(0.0, 10.0)],

        # diffusion parameters
        "tke_ed_coeff" => [bounded(0.01, 1.0)],
        "tke_diss_coeff" => [bounded(0.01, 1.0)],
        "static_stab_coeff" => [bounded(0.01, 1.0)],
        "tke_surf_scale" => [bounded(1.0, 16.0)],
        "Prandtl_number_0" =>[bounded(0.5, 1.5)],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [bounded(0.0, 100.0)],
        "pressure_normalmode_buoy_coeff1" => [bounded(0.0, 10.0)],
        "pressure_normalmode_drag_coeff" => [bounded(0.0, 50.0)],

        # surface
        "surface_area" => [bounded(0.01, 0.5)],

        # microphysics parameters
        "τ_acnv_rai" => [bounded(1e2, 1e4)],
        "τ_acnv_sno" => [bounded(1e1, 1e3)],
        "q_liq_threshold" => [bounded(1e-4, 1e-2)],
        "q_ice_threshold" => [bounded(1e-7,1e-5)],
        "microph_scaling" => [bounded(0.0, 10.0)],
        "microph_scaling_dep_sub" => [bounded(0.0, 10.0)],
        "microph_scaling_melt" => [bounded(0.0, 10.0)],
        "E_liq_rai" => [bounded(0.0, 10.0)],
        "E_liq_sno" => [bounded(0.0, 10.0)],
        "E_ice_rai" => [bounded(0.0, 10.0)],
        "E_ice_sno" => [bounded(0.0, 10.0)],
        "E_rai_sno" => [bounded(0.0, 10.0)],
    )

    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # entrainment parameters
        "entrainment_factor" => [0.13],
        "detrainment_factor" => [0.51],
        "entrainment_smin_tke_coeff" => [0.3],
        "updraft_mixing_frac" => [0.25],
        "entrainment_scale" => [4.0e-4],
        "sorting_power" => [2.0],
        "turbulent_entrainment_factor" => [0.075],

        # diffusion parameters
        "tke_ed_coeff" => [0.14],
        "tke_diss_coeff" => [0.22],
        "static_stab_coeff" => [0.4],
        "tke_surf_scale" => [3.75],
        "Prandtl_number_0" => [0.74],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [0.001],
        "pressure_normalmode_buoy_coeff1" => [0.12],
        "pressure_normalmode_drag_coeff" => [10.0],

        # surface
        "surface_area" => [0.1],

        # microphysics parameters
        "τ_acnv_rai" => [2500.0],
        "τ_acnv_sno" => [100.0],
        "q_liq_threshold" => [0.5e-3],
        "q_ice_threshold" => [1.0e-6],
        "microph_scaling" => [1.0],
        "microph_scaling_dep_sub" => [1.0],
        "microph_scaling_melt" => [1.0],
        "E_liq_rai" => [0.8],
        "E_liq_sno" => [0.1],
        "E_ice_rai" => [1.0],
        "E_ice_sno" => [0.1],
        "E_rai_sno" => [0.8],
    )

    config["unconstrained_σ"] = 1.0
    # Tight initial prior for Unscented
    # config["unconstrained_σ"] = 0.25
    return config
end

function get_scm_config()
    config = Dict()
    config["namelist_args"] = [
        ("time_stepping", "adapt_dt", true),
        ("time_stepping", "dt_max", 3.0),
        ("time_stepping", "dt_min", 1.0),
        ("stats_io", "frequency", 60.0),
        ("turbulence", "EDMF_PrognosticTKE", "updraft_number", 1),
        # ("turbulence", "EDMF_PrognosticTKE", "env_buoy_grad", "quadratures"),
        # ("turbulence", "EDMF_PrognosticTKE", "tke_diss_coeff", 0.254866694310963),
        # ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_buoy_coeff1", 0.14898301311928824),
        # ("turbulence", "EDMF_PrognosticTKE", "updraft_mixing_frac", 0.2019234651801264),
        # ("turbulence", "EDMF_PrognosticTKE", "Prandtl_number_0", 0.797717361562768),
        # ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_adv_coeff", 0.0009975725552479882),
        # ("turbulence", "EDMF_PrognosticTKE", "entrainment_smin_tke_coeff", 0.2399928217115579),
        # ("turbulence", "EDMF_PrognosticTKE", "tke_surf_scale", 3.7469796330314393),
        # ("turbulence", "EDMF_PrognosticTKE", "surface_area", 0.10090917770628326),
        # ("turbulence", "EDMF_PrognosticTKE", "detrainment_factor", 0.5252335519632857),
        # ("turbulence", "EDMF_PrognosticTKE", "static_stab_coeff", 0.3873437163848471),
        # ("turbulence", "EDMF_PrognosticTKE", "sorting_power", 2.4210446885173393),
        # ("turbulence", "EDMF_PrognosticTKE", "entrainment_factor", 0.10104274583196207),
        # ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_drag_coeff", 9.470550947450384),
        # ("turbulence", "EDMF_PrognosticTKE", "tke_ed_coeff", 0.17069970482575572),
        # ("turbulence", "EDMF_PrognosticTKE", "turbulent_entrainment_factor", 0.07395526189334647),
        # ("turbulence", "EDMF_PrognosticTKE", "entrainment_scale", 0.0004611728518427933),
        ("thermodynamics", "sgs", "quadrature"),
        ("thermodynamics", "quadrature_order", 3),
        ("thermodynamics", "quadrature_type", "gaussian"), #"gaussian" "log-normal"
    ]
    return config
end
