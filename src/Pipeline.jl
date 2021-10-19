module Pipeline

using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage

export init_calibration


"""
    init_calibration(N_ens::Int, N_iter::Int, job_id::String, config::Dict{Any, Any})

Initializes a calibration process given a configuration.

    Inputs:
    - N_ens :: Number of ensemble members.
    - N_iter :: Number of iterations.
    - job_id :: Unique job identifier for sbatch communication.
    - config :: User-defined configuration dictionary.
"""
function init_calibration(N_ens::Int, N_iter::Int, job_id::String, config::Dict{Any, Any})

    reference_type = config["reference"]["reference_type"]
    perform_PCA = config["regularization"]["perform_PCA"]
    normalize = config["regularization"]["normalize"]
    tikhonov_noise = config["regularization"]["tikhonov_noise"]

    save_eki_data = config["output"]["save_eki_data"]
    save_ensemble_data = config["output"]["save_ensemble_data"]
    overwrite_scm_file = config["output"]["overwrite_scm_file"]
    outdir_root = config["output"]["outdir_root"]

    algo = config["process"]["algorithm"]
    Δt = config["process"]["Δt"]

    params = config["prior"]["constraints"]

    kwargs_ref_model = Dict(
        :y_names => config["reference"]["y_names"],
        # Reference path specification
        :les_dir => config["reference"]["les_dir"],
        :scm_parent_dir => config["reference"]["scm_parent_dir"],
        :scm_suffix => config["reference"]["scm_suffix"],
        # Define observation window (s)
        :t_start => config["reference"]["t_start"],
        :t_end => config["reference"]["t_end"],
    )
    ref_models = construct_reference_models(config["reference"]["case_name"], kwargs_ref_model)

    # Create input scm stats and namelist file if files don't already exist
    run_SCM(ref_models, overwrite = overwrite_scm_file)
    # Generate reference statistics
    ref_stats = ReferenceStatistics(ref_models, reference_type, perform_PCA, normalize, tikhonov_noise = tikhonov_noise)

    algo_type = typeof(algo) == Sampler{Float64} ? "eks" : "eki"
    n_param = length(collect(keys(params)))
    d = length(ref_stats.y)
    outdir_path =
        joinpath(outdir_root, "results_$(algo_type)_dt$(Δt)_p$(n_param)_e$(N_ens)_i$(N_iter)_d$(d)_$(reference_type)")
    println("Name of outdir path for this EKP is: $outdir_path")
    mkpath(outdir_path)
    open("$(job_id).txt", "w") do io
        write(io, "$(outdir_path)\n")
    end
    priors = construct_priors(params, outdir_path = outdir_path, unconstrained_σ = config["prior"]["unconstrained_σ"])
    # parameters are sampled in unconstrained space
    initial_params = construct_initial_ensemble(priors, N_ens, rng_seed = rand(1:1000))
    generate_ekp(initial_params, ref_stats, algo, outdir_path = outdir_path)
    params_cons_i = transform_unconstrained_to_constrained(priors, initial_params)
    params = [c[:] for c in eachcol(params_cons_i)]
    versions = map(param -> generate_scm_input(param, get_name(priors), ref_models, ref_stats, outdir_path), params)

    # Store version identifiers for this ensemble in a common file
    write_versions(versions, 1, outdir_path = outdir_path)
end


end # module
