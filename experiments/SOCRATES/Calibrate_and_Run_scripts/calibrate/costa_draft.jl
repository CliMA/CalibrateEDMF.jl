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
import CalibrateEDMF.LESUtils: get_shallow_LES_library
# Import EKP modules
using JLD2
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Localizers
using EnsembleKalmanProcesses.ParameterDistributions

using TurbulenceConvection
const TC = TurbulenceConvection

# Cases defined as structs for quick access to default configs
struct SCT3Train end

NUM_NN_PARAMS = 12
SCM_RUN_TIME_HR = 72.0
T_START_HR = 60.0

LES_LENGTH_HR = 6.0 * 24

namelist_args = [
    ("time_stepping", "dt_min", 0.5),
    ("time_stepping", "dt_max", 5.0),
    ("stats_io", "frequency", 60.0),
    ("time_stepping", "t_max", 3600.0 * SCM_RUN_TIME_HR),
    ("t_interval_from_end_s", 3600.0 * SCM_RUN_TIME_HR),
    ("thermodynamics", "sgs", "quadrature"),
    ("turbulence", "EDMF_PrognosticTKE", "surface_area_bc", "Prognostic"),
    # ("turbulence", "EDMF_PrognosticTKE", "surface_area_bc", "Fixed"),
    # Add namelist_args defining entrainment closure, e.g.
    ("turbulence", "EDMF_PrognosticTKE", "entrainment_type", "total_rate"),
    ("turbulence", "EDMF_PrognosticTKE", "entr_dim_scale", "w_height"),
    ("turbulence", "EDMF_PrognosticTKE", "detr_dim_scale", "mf_grad"),
    ("turbulence", "EDMF_PrognosticTKE", "turbulent_entrainment_factor", 0.0),
    ("turbulence", "EDMF_PrognosticTKE", "entrainment", "None"),
    ("turbulence", "EDMF_PrognosticTKE", "ml_entrainment", "Linear"), 
    ("turbulence", "EDMF_PrognosticTKE", "min_area", 1e-10),
    ("turbulence", "EDMF_PrognosticTKE", "limit_min_area", false),
    ("turbulence", "EDMF_PrognosticTKE", "area_limiter_scale", 0.0), # if having problems with area limiter, try removing this line
    ("turbulence", "EDMF_PrognosticTKE", "entr_pi_subset", (1, 2, 3, 4, 6)),
    # ("turbulence", "EDMF_PrognosticTKE", "pi_norm_consts", [1e6, 1e3, 1.0, 1.0, 1.0, 1.0]),
    ("turbulence", "EDMF_PrognosticTKE", "pi_norm_consts", [100.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
    ("turbulence", "EDMF_PrognosticTKE", "linear_ent_params", zeros(NUM_NN_PARAMS)),
    ("turbulence", "EDMF_PrognosticTKE", "linear_ent_biases", true),
]

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(SCT3Train())
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

    config["obs_var_scaling"] = Dict("ql_mean" => 0.1,  "total_flux_s" => 0.1, "total_flux_qt"=> 0.1)

    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 12
    config["N_ens"] = 300 # Must be 2p+1 when algorithm is "Unscented"
    config["algorithm"] = "Inversion" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    # config["Δt"] = 1.0
    config["scheduler"] = DataMisfitController(on_terminate = "continue")
    # config["accelerator"] = NesterovAccelerator()
    config["accelerator"] = DefaultAccelerator()
    # config["additive_inflation"] = true
    # config["inflation_scaling_factor"] = 1/30.0
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = false
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    # https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/src/Localizers.jl#L63
    # use localizer when number of parameters > number of ensemble members
    # config["localizer"] = SEC(0.5, 0.1) # First arg is strength of localization, second is the minimum correlation retained
    return config
end

function get_reference_config(::SCT3Train)
    config = Dict()
    les_library = get_shallow_LES_library()

    # AMIP4K data: July, NE Pacific
    cfsite_numbers = (17, 18, 19, 22, 23)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 7, experiment = "amip")
    ref_dirs = [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers]

    # # AMIP4K data: October, NE Pacific
    cfsite_numbers = (17, 18, 19, 22, 23)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 10, experiment = "amip")
    append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])

    # # # AMIP4K data: January, SE Pacific
    # cfsite_numbers = (4, 6, 11, 13, 14)
    # les_kwargs = (forcing_model = "HadGEM2-A", month = 1, experiment = "amip")
    # append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])

    n_repeat = length(ref_dirs)

    config["case_name"] = repeat(["LES_driven_SCM"], n_repeat)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] =
        repeat([["s_mean", "ql_mean", "qt_mean", "total_flux_qt", "total_flux_s", "lwp_mean"]], n_repeat)
    config["y_dir"] = ref_dirs
    config["t_start"] = repeat([T_START_HR * 3600], n_repeat)
    config["t_end"] = repeat([SCM_RUN_TIME_HR * 3600], n_repeat)
    # Use full LES timeseries for covariance
    config["Σ_t_start"] = repeat([-1 * (LES_LENGTH_HR - SCM_RUN_TIME_HR - 24.0) * 3600.0], n_repeat) # Don't compute covariances for the first 24 hours of the LES simulation
    config["Σ_t_end"] = repeat([SCM_RUN_TIME_HR * 3600], n_repeat)
    config["time_shift"] = SCM_RUN_TIME_HR * 3600.0
    # config["batch_size"] = batch 
    config["write_full_stats"] = false
    config["namelist_args"] = repeat([namelist_args], n_repeat)
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        "linear_ent_params" => [repeat([no_constraint()], NUM_NN_PARAMS)...],
        # entrainment parameters

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

        # "area_limiter_scale" => [bounded(1.0, 100.0)], # costa's priors for this if we want
        # "area_limiter_power" => [bounded(1.0, 50.0)],  # costa's priors for this... but this is optimized for 0.9, maybe try this or a wider prior and see that works for 0.3

        # "min_area_limiter_scale" => [bounded(1.0, 100.0)],
        # "min_area_limiter_power" => [bounded(1000.0, 5000.0)],
    )

    linear_entr_prior_mean = 0.5 .* (rand(Int(NUM_NN_PARAMS / 2)) .- 0.5) .+ 1.0
    linear_detr_prior_mean = 0.5 .* (rand(Int(NUM_NN_PARAMS / 2)) .- 0.5) .+ 1.0
    linear_entr_prior_mean[end] = 0.01
    linear_detr_prior_mean[end] = 0.01
    # linear_entr_prior_mean[Int(NUM_NN_PARAMS / 2)] = 1.0
    # linear_detr_prior_mean[end] = 1.0
    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # data-driven entrainment parameters

        "linear_ent_params" => cat(linear_entr_prior_mean, linear_detr_prior_mean, dims = 1),

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

        # "area_limiter_scale" => [3.0],
        # "area_limiter_power" => [15.0],
    
        # "min_area_limiter_scale" => [2.0],
        # "min_area_limiter_power" => [2000.0],
    )

    linear_entr_unc_sigma = repeat([5.0], Int(NUM_NN_PARAMS / 2))
    linear_detr_unc_sigma = repeat([5.0], Int(NUM_NN_PARAMS / 2))

    linear_entr_unc_sigma[end] = 0.5
    linear_detr_unc_sigma[end] = 0.5

    non_vec_sigma = 2.0
    config["unconstrained_σ"] = Dict(
        # data-driven entrainment parameters

        # "linear_ent_params" => repeat([2.0], NUM_NN_PARAMS),
        "linear_ent_params" => cat(linear_entr_unc_sigma, linear_detr_unc_sigma, dims = 1),

        # diffusion parameters
        "tke_ed_coeff" => [non_vec_sigma],
        "tke_diss_coeff" => [non_vec_sigma],
        "static_stab_coeff" => [non_vec_sigma],
        "tke_surf_scale" => [non_vec_sigma],
        "Prandtl_number_0" => [non_vec_sigma],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [non_vec_sigma],
        "pressure_normalmode_buoy_coeff1" => [non_vec_sigma],
        "pressure_normalmode_drag_coeff" => [non_vec_sigma],

        # "area_limiter_scale" => [1.0],
        # "area_limiter_power" => [1.0],

        # "min_area_limiter_scale" => [1.0],
        # "min_area_limiter_power" => [1.0],

    )
    # Tight initial prior for Unscented
    # config["unconstrained_σ"] = 0.25
    return config
end

function get_scm_config()
    config = Dict()
    config["namelist_args"] = namelist_args
    return config
end
