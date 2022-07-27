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

# Cases defined as structs for quick access to default configs
struct SCT3Train end
struct SCT3Val end

batch = 15 # Possible batch sizes are [1, 3, 5, 9, 15, 27, 45, 135]

dt_default = 135.0 / batch
dt = dt_default * 1.0 # Treat as hyperparameter

namelist_args = [
    ("time_stepping", "dt_min", 0.5),
    ("time_stepping", "dt_max", 5.0),
    ("stats_io", "frequency", 60.0),
    ("thermodynamics", "sgs", "quadrature"),
    # Add namelist_args defining entrainment closure, e.g.
    ("turbulence", "EDMF_PrognosticTKE", "area_limiter_power", 0.0),
    ("turbulence", "EDMF_PrognosticTKE", "entr_dim_scale", "none"),
    ("turbulence", "EDMF_PrognosticTKE", "entrainment", "NN"),
]

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(SCT3Train())
    # Define reference used for validation
    config["validation"] = get_reference_config(SCT3Val())
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
    config["Δt"] = dt
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = false
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    # https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/src/Localizers.jl#L63
    config["localizer"] = SEC(0.5, 0.1) # First arg is strength of localization, second is the minimum correlation retained
    return config
end

function get_reference_config(::SCT3Train)
    config = Dict()
    # Get all 135 shallow cases from Shen et al (2022)
    les_library = get_shallow_LES_library()

    ref_dirs = []
    for model in keys(les_library)
        for month in keys(les_library[model])
            cfsite_numbers = Tuple(les_library[model][month]["cfsite_numbers"])
            les_kwargs = (forcing_model = model, month = parse(Int, month), experiment = "amip")
            append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])
        end
    end
    n_repeat = length(ref_dirs)

    config["case_name"] = repeat(["LES_driven_SCM"], n_repeat)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] =
        repeat([["s_mean", "ql_mean", "qt_mean", "total_flux_qt", "total_flux_s", "lwp_mean"]], n_repeat)
    config["y_dir"] = ref_dirs
    config["t_start"] = repeat([3.0 * 3600], n_repeat)
    config["t_end"] = repeat([6.0 * 3600], n_repeat)
    # Use full LES timeseries for covariance
    config["Σ_t_start"] = repeat([-5.75 * 24 * 3600], n_repeat)
    config["Σ_t_end"] = repeat([6.0 * 3600], n_repeat)
    config["batch_size"] = batch # Possible batch sizes are [1, 3, 5, 9, 15, 27, 45, 135]
    config["write_full_stats"] = false
    config["namelist_args"] = repeat([namelist_args], n_repeat)
    return config
end

function get_reference_config(::SCT3Val)
    config = Dict()

    # AMIP4K data: July, NE Pacific
    cfsite_numbers = (17, 18, 20, 22, 23)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 7, experiment = "amip4K")
    ref_dirs = [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers]

    # AMIP4K data: October, NE Pacific
    cfsite_numbers = (17, 18, 20, 22, 23)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 10, experiment = "amip4K")
    append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])

    # AMIP4K data: January, SE Pacific
    cfsite_numbers = (2, 4, 6, 11, 14)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 1, experiment = "amip4K")
    append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])

    n_repeat = length(ref_dirs)
    config["case_name"] = repeat(["LES_driven_SCM"], n_repeat)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] =
        repeat([["s_mean", "ql_mean", "qt_mean", "total_flux_qt", "total_flux_s", "lwp_mean"]], n_repeat)
    config["y_dir"] = ref_dirs
    config["t_start"] = repeat([3.0 * 3600], n_repeat)
    config["t_end"] = repeat([6.0 * 3600], n_repeat)
    # Use full LES timeseries for covariance
    config["Σ_t_start"] = repeat([-5.75 * 24 * 3600], n_repeat)
    config["Σ_t_end"] = repeat([6.0 * 3600], n_repeat)
    config["write_full_stats"] = false
    config["namelist_args"] = repeat([namelist_args], n_repeat)
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # data-driven entrainment parameters
        "nn_ent_params" => [repeat([no_constraint()], 58)...],
    )

    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # data-driven entrainment parameters
        "nn_ent_params" => 0.1 .* (rand(58) .- 0.5),
    )

    config["unconstrained_σ"] = 1.0
    # Tight initial prior for Unscented
    # config["unconstrained_σ"] = 0.25
    return config
end

function get_scm_config()
    config = Dict()
    config["namelist_args"] = namelist_args
    return config
end
