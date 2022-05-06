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
struct GridSearchCases end

function get_config()
    config = Dict()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(GridSearchCases())
    # Define the SCM static configuration
    config["scm"] = get_scm_config()
    # grid search configuration
    config["grid_search"] = get_grid_search_config()
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

function get_reference_config(::GridSearchCases)
    config = Dict()
    config["case_name"] = ["TRMM_LBA", "Rico", "TRMM_LBA"]
    n_cases = length(config["case_name"])
    # Flag to indicate source of data (LES or SCM) for reference data and covariance
    config["y_reference_type"] = SCM()
    # config["Σ_reference_type"] = LES()
    # Fields to learn from during training
    config["y_names"] = repeat([["ql_mean", "qt_mean", "total_flux_qt"]], n_cases)
    # LES data can be stored as an Artifact and downloaded lazily
    # config["y_dir"] = [...]  # only needed for loss map computation
    # provide list of dirs if different from `y_dir`
    # config["Σ_dir"] = [...]
    config["t_start"] = repeat([100.0], n_cases)
    config["t_end"] = repeat([200.0], n_cases)
    config["scm_parent_dir"] = repeat(["scm_init"], n_cases)
    config["scm_suffix"] = repeat(["0"], n_cases)
    return config
end

function get_scm_config()
    config = Dict()
    # List of tuples like [("time_stepping", "dt_min", 1.0)], or nothing
    config["namelist_args"] = nothing
    config["namelist_args"] = [
        ("time_stepping", "t_max", 200.0),
        ("turbulence", "EDMF_PrognosticTKE", "entrainment", "moisture_deficit"),
        ("turbulence", "EDMF_PrognosticTKE", "stochastic_entrainment", "prognostic_noisy_relaxation_process"),
        ("time_stepping", "adapt_dt", false),
    ]
    return config
end

function get_grid_search_config()
    config = Dict()
    # Grid search is performed over each pair of parameters, across all specified values
    config["parameters"] = Dict(
        "general_stochastic_ent_params_{1}" => [0.3, 0.4],
        "general_stochastic_ent_params_{2}" => [0.2, 0.3],
        "entrainment_factor" => [0.1],
    )
    # Number of simulations to run with identical configuration (except random seed)
    config["ensemble_size"] = 1
    # grid search output data stored in `<output_root_dir>/output/YYmmdd_abc`
    # config["output_root_dir"] = pwd()  # provided explicitly in `grid_search_test.jl`
    # Perform grid search and loss map calculation for either the `reference` or `validation` set.
    config["sim_type"] = "reference"
    return config
end
