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

Initializes a calibration process given a configuration, and a pipeline mode.

    Inputs:
    - N_ens :: Number of ensemble members.
    - N_iter :: Number of iterations.
    - job_id :: Unique job identifier for sbatch communication.
    - config :: User-defined configuration dictionary.
    - mode :: Whether the calibration process is parallelized through HPC resources
      or using Julia's pmap.
"""
function init_calibration(
    N_ens::Int,
    N_iter::Int,
    config::Dict{Any, Any};
    mode::String = "hpc",
    job_id::String = "12345",
)
    @assert mode in ["hpc", "pmap"]

    y_ref_type = config["reference"]["y_reference_type"]
    Σ_ref_type = config["reference"]["Σ_reference_type"]

    perform_PCA = config["regularization"]["perform_PCA"]
    normalize = config["regularization"]["normalize"]
    tikhonov_noise = config["regularization"]["tikhonov_noise"]
    tikhonov_mode = config["regularization"]["tikhonov_mode"]
    variance_loss = config["regularization"]["variance_loss"]
    dim_scaling = config["regularization"]["dim_scaling"]

    save_eki_data = config["output"]["save_eki_data"]
    save_ensemble_data = config["output"]["save_ensemble_data"]
    overwrite_scm_file = config["output"]["overwrite_scm_file"]
    outdir_root = config["output"]["outdir_root"]

    algo_name = config["process"]["algorithm"]
    Δt = config["process"]["Δt"]

    params = config["prior"]["constraints"]

    n_cases = length(config["reference"]["case_name"])
    # if Σ_dir is `nothing`, it is expanded to an array of `nothing`
    Σ_dir = expand_dict_entry(config["reference"], "Σ_dir", n_cases)

    # Similarly, generate `Σ_t_start` and `Σ_t_end`
    Σ_t_start = expand_dict_entry(config["reference"], "Σ_t_start", n_cases)
    Σ_t_end = expand_dict_entry(config["reference"], "Σ_t_start", n_cases)

    # Construct reference models
    kwargs_ref_model = Dict(
        :y_names => config["reference"]["y_names"],
        # Reference path specification
        :y_dir => config["reference"]["y_dir"],
        :Σ_dir => Σ_dir,
        :scm_parent_dir => config["reference"]["scm_parent_dir"],
        :scm_suffix => config["reference"]["scm_suffix"],
        # Case name
        :case_name => config["reference"]["case_name"],
        # Define observation window (s)
        :t_start => config["reference"]["t_start"],
        :t_end => config["reference"]["t_end"],
        :Σ_t_start => Σ_t_start,
        :Σ_t_end => Σ_t_end,
    )
    ref_models = construct_reference_models(kwargs_ref_model)

    # Create input scm stats and namelist file if files don't already exist
    run_SCM(ref_models, overwrite = overwrite_scm_file)

    # Generate reference statistics
    ref_stats = ReferenceStatistics(
        ref_models,
        perform_PCA,
        normalize,
        tikhonov_noise = tikhonov_noise,
        tikhonov_mode = tikhonov_mode,
        variance_loss = variance_loss,
        dim_scaling = dim_scaling,
        y_type = y_ref_type,
        Σ_type = Σ_ref_type,
    )

    # Dimensionality
    n_param = length(collect(keys(params)))
    d = length(ref_stats.y)
    if algo_name == "Unscented"
        N_ens = 2 * n_param + 1
        @warn "Number of ensemble members overwritten to 2p + 1 for Unscented Kalman Inversion."
    end

    # Output path
    outdir_path = joinpath(
        outdir_root,
        "results_$(algo_name)_dt$(Δt)_p$(n_param)_e$(N_ens)_i$(N_iter)_d$(d)_$(typeof(y_ref_type))",
    )
    @info "Name of outdir path for this EKP is: $outdir_path"
    mkpath(outdir_path)

    priors = construct_priors(params, outdir_path = outdir_path, unconstrained_σ = config["prior"]["unconstrained_σ"])
    # parameters are sampled in unconstrained space
    if algo_name == "Inversion" || algo_name == "Sampler"
        algo = algo_name == "Inversion" ? Inversion() : Sampler(vcat(get_mean(priors)...), get_cov(priors))
        initial_params = construct_initial_ensemble(priors, N_ens, rng_seed = rand(1:1000))
        ekobj = generate_ekp(initial_params, ref_stats, algo, outdir_path = outdir_path)
    elseif algo_name == "Unscented"
        algo = Unscented(vcat(get_mean(priors)...), get_cov(priors), 1.0, 0)
        ekobj = generate_ekp(ref_stats, algo, outdir_path = outdir_path)
    end

    if mode == "hpc"
        open("$(job_id).txt", "w") do io
            write(io, "$(outdir_path)\n")
        end
        params_cons_i = transform_unconstrained_to_constrained(priors, initial_params)
        params = [c[:] for c in eachcol(params_cons_i)]
        versions = map(param -> generate_scm_input(param, get_name(priors), ref_models, ref_stats, outdir_path), params)
        # Store version identifiers for this ensemble in a common file
        write_versions(versions, 1, outdir_path = outdir_path)

    elseif mode == "pmap"
        return Dict(
            "ekobj" => ekobj,
            "priors" => priors,
            "ref_stats" => ref_stats,
            "ref_models" => ref_models,
            "d" => d,
            "n_param" => n_param,
            "outdir_path" => outdir_path,
        )
    end
end


end # module
