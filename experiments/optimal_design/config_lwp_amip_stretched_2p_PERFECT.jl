#= Custom calibration configuration file. =#
using Glob
using Distributions
using StatsBase
using LinearAlgebra
using Random
using JLD2
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

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

# Cases defined as structs for quick access to default configs
struct ShallowHadGEM end
struct ShallowHadGEMVal end

# No batching for now
batch = 42 # Possible batch sizes are [1, 2, 4, 19, 38, and 76]

# Hyperparameters
dt_default = 42.0 / batch
dt = dt_default * 1.0
l2_reg_scale = 1.0

# TC custom configuration
namelist_args = [
    ("time_stepping", "dt_min", 0.5),
    ("time_stepping", "dt_max", 5.0),
    ("stats_io", "frequency", 60.0),
    ("thermodynamics", "sgs", "quadrature"),
    ("grid", "stretch", "flag", true),
    ("grid", "stretch", "nz", 55),
    ("grid", "stretch", "dz_surf", 30.0),
    ("grid", "stretch", "dz_toa", 8000.0),
    ("grid", "stretch", "z_toa", 45000.0),
]

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(ShallowHadGEM())
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
    config["perform_PCA"] = false # Performs PCA on data
    config["variance_loss"] = 1.0e-2 # Variance truncation level in PCA
    config["normalize"] = true  # whether to normalize data by pooled variance
    config["tikhonov_mode"] = "relative" # Tikhonov regularization
    config["tikhonov_noise"] = 1.0e-6 # Tikhonov regularization
    config["dim_scaling"] = true # Dimensional scaling of the loss

    # Parameter regularization: L2 regularization with respect to prior mean
    # Set to `nothing` or `0.0` to use prior covariance as regularizer.
    # To turn off regularization, set config["process"]["augmented"] to false.
    config["l2_reg"] = l2_reg_scale / dt
    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 10
    config["N_ens"] = 50 #33 # Must be 2p+1 when algorithm is "Unscented"
    config["algorithm"] = "Inversion" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = dt
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = false
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    return config
end

function get_reference_config(::ShallowHadGEM)
    config = Dict()
    # Get all 135 shallow cases from Shen et al (2022)
    les_library = get_shallow_LES_library()

    # Get LES directories for noise construction. For now, only use HadGEM2-A model
    les_ref_dirs = []
    for model in ["HadGEM2-A"] # keys(les_library)
	    for month in ["10", "07"]#keys(les_library[model])
            cfsite_numbers = Tuple(les_library[model][month]["cfsite_numbers"])
            les_kwargs = (forcing_model = model, month = parse(Int, month), experiment = "amip")
            append!(
                les_ref_dirs,
                [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers],
            )
        end
    end
    n_repeat = length(les_ref_dirs)

    # Get SCM directories for truth values (perfect model experiment)
    perf_model_dir = "/groups/esm/ilopezgo/optimal_design/perf_model_HadGEM2_nz55_B38_Inv_d40_p2_ewR"
    perf_model_exps = Glob.glob(relpath(abspath(joinpath(perf_model_dir, "Output*"))))
    # Order 1-to-1 with LES directories
    ordered_perf_model_exps = []
    for ref_dir in les_ref_dirs
        for perf_model_exp in perf_model_exps
            id_str = join(split(ref_dir, ".")[2:3], ".")
            if occursin(id_str, perf_model_exp)
                push!(ordered_perf_model_exps, perf_model_exp)
                break
            end
        end
    end

    config["case_name"] = repeat(["LES_driven_SCM"], n_repeat)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = SCM()
    config["Σ_reference_type"] = LES() # Keep variance from LES
    config["y_names"] = repeat([["lwp_mean"]], n_repeat)
    # write a generator function `les_to_scm_names`
    config["y_dir"] = ordered_perf_model_exps
    config["Σ_dir"] = les_ref_dirs
    config["t_start"] = repeat([3.0 * 3600], n_repeat)
    config["t_end"] = repeat([6.0 * 3600], n_repeat)
    # Use full LES timeseries for covariance
    config["Σ_t_start"] = repeat([-5.75 * 24 * 3600], n_repeat)
    config["Σ_t_end"] = repeat([6.0 * 3600], n_repeat)
    config["batch_size"] = batch # Possible batch sizes are [1, 2, 4, 19, 38, and 76]
    config["write_full_stats"] = true
    config["namelist_args"] = repeat([namelist_args], n_repeat)
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # entrainment parameters
        "entrainment_factor" => [bounded(0.0, 1.0)],
        "detrainment_factor" => [bounded(0.0, 1.0)],
    )

    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # entrainment parameters
        "entrainment_factor" => [0.13], # Truth: [0.08758892747212922], # From previous calibration to LES
        "detrainment_factor" => [0.51], # Truth: [0.699804880392696],   # From previous calibration to LES
    )
    config["unconstrained_σ"] = 0.75
    return config
end

function get_scm_config()
    config = Dict()
    config["namelist_args"] = namelist_args
    return config
end
