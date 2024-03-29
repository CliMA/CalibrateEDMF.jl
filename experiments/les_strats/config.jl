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
struct LesDrivenScm end
struct LesDrivenScmVal end
# struct MyAwesomeSetup end

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(LesDrivenScm())
    # Define reference used for validation
    config["validation"] = get_reference_config(LesDrivenScmVal())
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
    config["l2_reg"] = nothing
    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 20
    config["N_ens"] = 13
    config["algorithm"] = "Unscented" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = 1.0
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = false
    config["failure_handler"] = "sample_succ_gauss"
    return config
end

function get_reference_config(::ObsCampaigns)
    config = Dict()
    config["case_name"] = ["DYCOMS_RF01", "GABLS", "Bomex"]
    # Flag to indicate source of data (LES or SCM) for reference data and covariance
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] = [
        ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"],
        ["thetal_mean", "u_mean", "v_mean", "tke_mean"],
        ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"],
    ]
    config["y_dir"] = [
        "/groups/esm/ilopezgo/Output.DYCOMS_RF01.may20",
        "/groups/esm/ilopezgo/Output.GABLS.iles128wCov",
        "/groups/esm/ilopezgo/Output.Bomex.may18",
    ]
    # provide list of dirs if different from `y_dir`
    # config["Σ_dir"] = [...]
    config["t_start"] = [2, 7, 4] * 3600.0
    config["t_end"] = [4, 9, 6] * 3600.0
    # Specify averaging intervals for covariance, if different from mean vector (`t_start` & `t_end`)
    # config["Σ_t_start"] = [...]
    # config["Σ_t_end"] = [...]
    return config
end

function get_reference_config(::LesDrivenScm)
    config = Dict()

    # AMIP data: July
    cfsite_numbers = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 7, experiment = "amip")
    ref_dirs = [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers]
    # AMIP data: October
    cfsite_numbers = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 10, experiment = "amip")
    append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])
    # AMIP4K data: July
    cfsite_numbers = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 7, experiment = "amip4K")
    append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])
    # AMIP4K data: October
    cfsite_numbers = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 10, experiment = "amip4K")
    append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])

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
    config["batch_size"] = 5
    config["write_full_stats"] = false
    return config
end

function get_reference_config(::LesDrivenScmVal)
    config = Dict()
    cfsite_numbers = (2, 4, 6, 8, 10, 12, 14, 18, 20, 22)
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
    # config["batch_size"] = 5
    config["write_full_stats"] = true
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # entrainment parameters
        "entrainment_factor" => [bounded(0.0, 1.0)],
        "detrainment_factor" => [bounded(0.0, 1.0)],
        "sorting_power" => [bounded(0.0, 4.0)],
        # "turbulent_entrainment_factor" => [bounded(0.0, 0.03)],
        # "entrainment_sigma" => [bounded(5.0, 15.0)],
        # "entrainment_scale" => [bounded(0.001, 0.007)],

        # diffusion parameters
        # "tke_ed_coeff" => [bounded(0.01, 0.3)],
        # "tke_diss_coeff" => [bounded(0.01, 0.45)],
        # "static_stab_coeff" => [bounded(0.1, 0.7)],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [bounded(0.0, 1.0)],
        "pressure_normalmode_buoy_coeff1" => [bounded(0.0, 1.0)],
        "pressure_normalmode_drag_coeff" => [bounded(0.0, 50.0)],
    )
    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # entrainment parameters
        "entrainment_factor" => [0.13],
        "detrainment_factor" => [0.51],
        "sorting_power" => [2.0],
        # "turbulent_entrainment_factor" => [0.015],
        # "entrainment_sigma" => [10.0],
        # "entrainment_scale" => [0.004],

        # diffusion parameters
        # "tke_ed_coeff" => [0.14],
        # "tke_diss_coeff" => [0.22],
        # "static_stab_coeff" => [0.4],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [0.1],
        "pressure_normalmode_buoy_coeff1" => [0.12],
        "pressure_normalmode_drag_coeff" => [10.0],
    )
    # Worse prior to test training
    # config["prior_mean"] = Dict(
    # entrainment parameters
    # "entrainment_factor" => 0.4,
    # "detrainment_factor" => 0.7,
    # "sorting_power" => 1.0,
    # "turbulent_entrainment_factor" => 0.015,
    # "entrainment_sigma" => 10.0,
    # "entrainment_scale" => 0.004,

    # diffusion parameters
    # "tke_ed_coeff" => 0.14,
    # "tke_diss_coeff" => 0.22,
    # "static_stab_coeff" => 0.4,

    # momentum exchange parameters
    # "pressure_normalmode_adv_coeff" => 0.5,
    # "pressure_normalmode_buoy_coeff1" => 0.5,
    # "pressure_normalmode_drag_coeff" => 25.0,
    # )

    # config["unconstrained_σ"] = 1.0
    # Tight initial prior for Unscented
    config["unconstrained_σ"] = 0.1
    return config
end

function get_scm_config()
    config = Dict()
    config["namelist_args"] = [
        ("time_stepping", "dt_min", 2.0),
        ("time_stepping", "dt_max", 10.0),
        ("stats_io", "frequency", 60.0),
        ("grid", "nz", 80),
    ]
    return config
end
