module Pipeline

using Random
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.NetCDFIO
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage

export init_calibration


"""
    init_calibration(job_id::String, config::Dict{Any, Any})

Initializes a calibration process given a configuration, and a pipeline mode.

    Inputs:
    - job_id :: Unique job identifier for sbatch communication.
    - config :: User-defined configuration dictionary.
    - mode :: Whether the calibration process is parallelized through HPC resources
      or using Julia's pmap.
"""
function init_calibration(config::Dict{Any, Any}; mode::String = "hpc", job_id::String = "12345", config_path = nothing)
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

    N_ens = config["process"]["N_ens"]
    N_iter = config["process"]["N_iter"]
    algo_name = config["process"]["algorithm"]
    Δt = config["process"]["Δt"]

    params = config["prior"]["constraints"]

    n_cases = length(config["reference"]["case_name"])
    # if Σ_dir is `nothing`, it is expanded to an array of `nothing`
    Σ_dir = expand_dict_entry(config["reference"], "Σ_dir", n_cases)

    # Similarly, generate `Σ_t_start` and `Σ_t_end`
    Σ_t_start = expand_dict_entry(config["reference"], "Σ_t_start", n_cases)
    Σ_t_end = expand_dict_entry(config["reference"], "Σ_t_end", n_cases)

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
        variance_loss = variance_loss,
        tikhonov_noise = tikhonov_noise,
        tikhonov_mode = tikhonov_mode,
        dim_scaling = dim_scaling,
        y_type = y_ref_type,
        Σ_type = Σ_ref_type,
    )

    # Dimensionality
    n_param = sum(map(length, collect(values(params))))
    d = length(ref_stats.y)
    if algo_name == "Unscented"
        N_ens = 2 * n_param + 1
        @warn "Number of ensemble members overwritten to 2p + 1 for Unscented Kalman Inversion."
    end

    # Output path
    outdir_path = joinpath(
        outdir_root,
        "results_$(algo_name)_dt$(Δt)_p$(n_param)_e$(N_ens)_i$(N_iter)_d$(d)_$(typeof(y_ref_type))_$(rand(11111:99999))",
    )
    @info "Name of outdir path for this EKP is: $outdir_path"
    mkpath(outdir_path)
    if !isnothing(config_path)
        cp(config_path, joinpath(outdir_path, "config.jl"))
    end

    priors = construct_priors(params, outdir_path = outdir_path, unconstrained_σ = config["prior"]["unconstrained_σ"])
    # parameters are sampled in unconstrained space
    if algo_name == "Inversion" || algo_name == "Sampler"
        algo = algo_name == "Inversion" ? Inversion() : Sampler(vcat(get_mean(priors)...), get_cov(priors))
        initial_params = construct_initial_ensemble(priors, N_ens, rng_seed = rand(1:1000))
        ekobj = generate_ekp(initial_params, ref_stats, algo, outdir_path = outdir_path)
    elseif algo_name == "Unscented"
        algo = Unscented(vcat(get_mean(priors)...), get_cov(priors), 1.0, 1)
        ekobj = generate_ekp(ref_stats, algo, outdir_path = outdir_path)
    end

    # Diagnostics IO
    init_diagnostics(config, outdir_path, ref_stats, ref_models, ekobj, priors)

    if mode == "hpc"
        open("$(job_id).txt", "w") do io
            write(io, "$(outdir_path)\n")
        end
        params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekobj))
        params = [c[:] for c in eachcol(params_cons_i)]
        mod_evaluators = [ModelEvaluator(param, get_name(priors), ref_models, ref_stats) for param in params]
        versions = map(mod_eval -> generate_scm_input(mod_eval, outdir_path), mod_evaluators)
        # Store version identifiers for this ensemble in a common file
        write_versions(versions, 1, outdir_path = outdir_path)

    elseif mode == "pmap"
        return Dict(
            "ekobj" => ekobj,
            "priors" => priors,
            "ref_stats" => ref_stats,
            "ref_models" => ref_models,
            "outdir_path" => outdir_path,
        )
    end
end

"""
    init_diagnostics(
        config::Dict{Any, Any},
        outdir_path::String,
        ref_stats::ReferenceStatistics,
        ekp::EnsembleKalmanProcess,
        priors::ParameterDistribution,
    )

Creates a diagnostics netcdf file.

    Inputs:
    - config :: User-defined configuration dictionary.
    - outdir_path :: Path of results directory.
    - ref_stats :: ReferenceStatistics.
    - ekp :: Initial EnsembleKalmanProcess, containing parameter information,
     but no forward model evaluations.
    - priors:: Prior distributions of the parameters.
"""
function init_diagnostics(
    config::Dict{Any, Any},
    outdir_path::String,
    ref_stats::ReferenceStatistics,
    ref_models::Vector{ReferenceModel},
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
)
    diags = NetCDFIO_Diags(config, outdir_path, ref_stats, ekp)
    # Write reference
    io_reference(diags, ref_stats, ref_models)
    # Add metric fields
    init_metrics(diags)
    # Add diags, write first ensemble diags
    open_files(diags)
    init_iteration_io(diags)
    init_particle_diags(diags, ekp, priors)
    close_files(diags)
end

end # module
