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
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
using JLD2

# Cases defined as structs for quick access to default configs
struct ObsCampaigns end
struct LesDrivenScm end
# struct MyAwesomeSetup end

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define preconditioning and regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(LesDrivenScm())
    # Define the parameter priors
    config["prior"] = get_prior_config()
    # Define the kalman process
    config["process"] = get_process_config()
    return config
end

function get_output_config()
    config = Dict()
    config["outdir_root"] = pwd()
    config["save_eki_data"] = true  # eki output
    config["save_ensemble_data"] = false  # .nc-files from each ensemble run
    config["overwrite_scm_file"] = false # Flag for overwritting SCM input file
    return config
end

function get_regularization_config()
    config = Dict()
    config["perform_PCA"] = true # Performs PCA on data
    config["variance_loss"] = 1.0e-2 # Variance truncation level in PCA
    config["normalize"] = true  # whether to normalize data by pooled variance
    config["tikhonov_mode"] = "relative" # Tikhonov regularization
    config["tikhonov_noise"] = 10.0 # Tikhonov regularization
    config["dim_scaling"] = true # Tikhonov regularization
    config["precondition"] = true # Application of prior preconditioning
    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 4
    config["N_ens"] = 5
    config["algorithm"] = "Unscented" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = true # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    config["Δt"] = 1.0 # Artificial time stepper of the EKI.
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
    config["scm_suffix"] = repeat(["000000"], length(config["case_name"]))
    config["scm_parent_dir"] = repeat(["scm_init"], length(config["case_name"]))
    config["t_start"] = [2, 7, 4] * 3600.0
    config["t_end"] = [4, 9, 6] * 3600.0
    # Specify averaging intervals for covariance, if different from mean vector (`t_start` & `t_end`)
    # config["Σ_t_start"] = [...]
    # config["Σ_t_end"] = [...]
    return config
end


function get_reference_config(::LesDrivenScm)
    config = Dict()
    config["case_name"] = repeat(["LES_driven_SCM"], 3)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = LES()
    config["Σ_reference_type"] = LES()
    config["y_names"] = repeat([["thetal_mean", "ql_mean", "qt_mean", "s_mean", "s_mean"]], 3)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 7, experiment = "amip")
    config["y_dir"] = [
        get_cfsite_les_dir(17; les_kwargs...),
        get_cfsite_les_dir(19; les_kwargs...),
        get_cfsite_les_dir(22; les_kwargs...),
    ]
    config["scm_suffix"] =
        [get_gcm_les_uuid(17; les_kwargs...), get_gcm_les_uuid(19; les_kwargs...), get_gcm_les_uuid(22; les_kwargs...)]
    config["scm_parent_dir"] = repeat(["scm_init"], 3)
    config["t_start"] = repeat([9.0 * 3600], 3)
    config["t_end"] = repeat([12.0 * 3600], 3)
    # config["Σ_t_start"] = [...]
    # config["Σ_t_end"] = [...]
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # entrainment parameters
        "entrainment_factor" => [bounded(0.01, 0.3)],
        "detrainment_factor" => [bounded(0.01, 0.9)],
        "sorting_power" => [bounded(0.25, 4.0)],
        "tke_ed_coeff" => [bounded(0.01, 0.5)],
        "tke_diss_coeff" => [bounded(0.01, 0.5)],
        "pressure_normalmode_adv_coeff" => [bounded(0.0, 0.5)],
        "pressure_normalmode_buoy_coeff1" => [bounded(0.0, 0.5)],
        "pressure_normalmode_drag_coeff" => [bounded(5.0, 15.0)],
        "static_stab_coeff" => [bounded(0.1, 0.8)],
    )
    config["unconstrained_σ"] = 0.5
    return config
end
