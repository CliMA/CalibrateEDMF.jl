#= Custom calibration configuration file. =#

using Distributions
using StatsBase
using LinearAlgebra
using Random
using CalibrateEDMF
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.LESUtils
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.HelperFuncs
# Import EKP modules
using JLD2
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

# Cases defined as structs for quick access to default configs
struct SOCRATES_Train end
struct SOCRATES_Val end

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(SOCRATES_Train())
    # Define reference used for validation
    config["validation"] = get_reference_config(SOCRATES_Val())
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

    # Parameter regularization: L2 regularization with respect to prior mean.
    #  - Set to `nothing` to use prior covariance as regularizer,
    #  - Set to a float for isotropic parameter regularization.
    #  - Pass a dictionary of lists similar to config["prior_mean"] for
    #       anisotropic regularization. The dictionary must be complete.
    #       If you want to avoid regularizing a certain parameter, set the entry
    #       to [0].
    # To turn off regularization, set config["process"]["augmented"] to false.
    #
    #
    # Defaults set to batch_size/total_size to match total dataset uncertainty
    # in UKI. Feel free to set treat these as hyperparameters.
    config["l2_reg"] = Dict(
        # entrainment parameters
        "nn_ent_params" => repeat([0.0], 58),
        "turbulent_entrainment_factor" => [5.0 / 60.0],

        # diffusion parameters
        "tke_ed_coeff" => [5.0 / 60.0],
        "tke_diss_coeff" => [5.0 / 60.0],
        "static_stab_coeff" => [5.0 / 60.0],
        "tke_surf_scale" => [5.0 / 60.0],
        "Prandtl_number_0" => [5.0 / 60.0],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [5.0 / 60.0],
        "pressure_normalmode_buoy_coeff1" => [5.0 / 60.0],
        "pressure_normalmode_drag_coeff" => [5.0 / 60.0],

        # surface
        "surface_area" => [5.0 / 60.0],
    )
    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 50
    config["N_ens"] = 50 # Must be 2p+1 when algorithm is "Unscented"
    config["algorithm"] = "Inversion" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = 60.0 / 5.0 # 1.0
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = true
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    return config
end

function get_reference_config(::SOCRATES_Train)
    config = Dict()

    # Setup SOCRATES run arguments I guess
    flight_numbers = (9,)
    aux_kwargs     = (,) # fill in later
    # append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])
    
    # need reference dirs from wherever i put my truth, maybe add a SOCRATESUtils to match les_utils etc.
    n_repeat = length(ref_dirs)
    config["case_name"] = repeat(["SOCRATES"], n_repeat)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = SOCRATES()
    config["Σ_reference_type"] = SOCRATES()
    config["y_names"] =
    config["y_dir"] = ref_dirs
    config["t_start"] = repeat([0], n_repeat)
    config["t_end"] = repeat([14.0 * 3600], n_repeat)
    # Use full  timeseries for covariance (for us we just use what)?
    config["Σ_t_start"] = repeat([-5.75 * 24 * 3600], n_repeat) # might need to discard beginning for spinup etc..
    config["Σ_t_end"] = repeat([6.0 * 3600], n_repeat)
    config["batch_size"] = 5 # 
    config["write_full_stats"] = false
    return config
end

function get_reference_config(::SOCRATES_Val)
    config = Dict()

    # Train on same thing? or waht do we do here        repeat([["s_mean", "ql_mean", "qt_mean", "total_flux_qt", "total_flux_s", "u_mean", "v_mean"]], n_repeat) # are these what we want to validate on? so theta_li, qt, for our dynamical calibration?

    flight_numbers = (9,) # just validate on same flight
    aux_kwargs     = (,) # fill in later
    # ref_dirs = [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers]

    n_repeat = length(ref_dirs)
    config["case_name"] = repeat(["SOCRATES"], n_repeat)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = SOCRATES()
    config["Σ_reference_type"] = SOCRATES()
    config["y_names"] =
        repeat([["s_mean", "ql_mean", "qt_mean", "total_flux_qt", "total_flux_s", "u_mean", "v_mean"]], n_repeat) # change here to qt, theta_li? (remove s for now or use theta_l for now since we have that)
    config["y_dir"] = ref_dirs
    config["t_start"] = repeat([0.0 * 3600], n_repeat)
    config["t_end"] = repeat([14.0 * 3600], n_repeat)
    # Use full LES timeseries for covariance (we will change to use what here, a timeseries from the les data? or what)
    # config["Σ_t_start"] = repeat([-5.75 * 24 * 3600], n_repeat)
    # config["Σ_t_end"] = repeat([6.0 * 3600], n_repeat)
    config["write_full_stats"] = false
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # entrainment parameters
        # "nn_ent_params" => [repeat([no_constraint()], 58)...], # remove
        # need to add entrainment parameters for moisture deficit clousure
        "turbulent_entrainment_factor" => [bounded(0.0, 10.0)],

        # diffusion parameters
        "tke_ed_coeff" => [bounded(0.01, 1.0)],
        "tke_diss_coeff" => [bounded(0.01, 1.0)],
        "static_stab_coeff" => [bounded(0.01, 1.0)],
        "tke_surf_scale" => [bounded(1.0, 16.0)],
        "Prandtl_number_0" => [bounded(0.5, 1.5)],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [bounded(0.0, 100.0)],
        "pressure_normalmode_buoy_coeff1" => [bounded(0.0, 10.0)],
        "pressure_normalmode_drag_coeff" => [bounded(0.0, 50.0)],

        # surface
        "surface_area" => [bounded(0.01, 0.5)],
    )

    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # entrainment parameters
        "nn_ent_params" => 0.1 .* (rand(58) .- 0.5),
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
    )

    config["unconstrained_σ"] = 1.0
    # Tight initial prior for Unscented
    # config["unconstrained_σ"] = 0.25
    return config
end

function get_scm_config() # set all my namelist stuff here
    config = Dict()
    config["namelist_args"] = [
        ("time_stepping", "dt_min", 1.0),
        ("time_stepping", "dt_max", 2.0),
        ("stats_io", "frequency", 60.0),
        ("turbulence", "EDMF_PrognosticTKE", "entrainment", "None"),
        ("turbulence", "EDMF_PrognosticTKE", "ml_entrainment", "NN"),
        ("turbulence", "EDMF_PrognosticTKE", "area_limiter_power", 0.0),
        ("turbulence", "EDMF_PrognosticTKE", "entr_dim_scale", "inv_z"),
        ("turbulence", "EDMF_PrognosticTKE", "detr_dim_scale", "inv_z"),
    ]
    return config
end

