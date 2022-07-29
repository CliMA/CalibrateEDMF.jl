module Pipeline

export init_calibration, ek_update, versioned_model_eval, restart_calibration

using Statistics
using Random
using JLD2
import Dates

# Import EKP modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
import EnsembleKalmanProcesses: update_ensemble!

using ..DistributionUtils
using ..ReferenceModels
using ..ReferenceStats
using ..TurbulenceConvectionUtils
using ..NetCDFIO
using ..HelperFuncs
using ..KalmanProcessUtils
using ..AbstractTypes
import ..AbstractTypes: Opt, OptVec, OptString

"""
    init_calibration(job_id::String, config::Dict{Any, Any})

Initializes a calibration process given a configuration, and a pipeline mode.

Inputs:

 - job_id :: Unique job identifier for sbatch communication.
 - config :: User-defined configuration dictionary.
 - mode :: Whether the calibration process is parallelized through HPC resources or using Julia's pmap.
"""
function init_calibration(config::Dict{Any, Any}; mode::String = "hpc", job_id::String = "12345", config_path = nothing)
    @info "Initialize calibration on $(Dates.now())"
    @assert mode in ["hpc", "pmap"]

    ref_config = config["reference"]
    y_ref_type = ref_config["y_reference_type"]
    batch_size = get_entry(ref_config, "batch_size", nothing)
    namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    kwargs_ref_model = get_ref_model_kwargs(ref_config; global_namelist_args = namelist_args)

    reg_config = config["regularization"]
    kwargs_ref_stats = get_ref_stats_kwargs(ref_config, reg_config)
    l2_reg = get_entry(reg_config, "l2_reg", nothing)

    out_config = config["output"]
    outdir_root = get_entry(out_config, "outdir_root", pwd())

    proc_config = config["process"]
    N_ens = proc_config["N_ens"]
    N_iter = proc_config["N_iter"]
    algo_name = get_entry(proc_config, "algorithm", "Inversion")

    Δt_scheduler = get_entry(proc_config, "Δt", 1.0)
    Δt = get_Δt(Δt_scheduler, 1)

    augmented = get_entry(proc_config, "augmented", false)
    failure_handler = get_entry(proc_config, "failure_handler", "high_loss")
    localizer = get_entry(proc_config, "localizer", NoLocalization())

    params = config["prior"]["constraints"]
    unc_σ = get_entry(config["prior"], "unconstrained_σ", 1.0)
    prior_μ = get_entry(config["prior"], "prior_mean", nothing)
    param_map = get_entry(config["prior"], "param_map", HelperFuncs.do_nothing_param_map())  # do-nothing param map by default

    val_config = get(config, "validation", nothing)

    # Dimensionality
    n_param = sum(map(length, collect(values(params))))
    if algo_name == "Unscented"
        @assert N_ens == 2 * n_param + 1 "Number of ensemble members must be 2p + 1 in Unscented Kalman Inversion."
    end

    # Minibatch mode
    if !isnothing(batch_size)
        @info "Training using mini-batches."
        ref_model_batch = ReferenceModelBatch(kwargs_ref_model)
        global_ref_models = deepcopy(ref_model_batch.ref_models)
        ref_models, batch_indices = get_minibatch!(ref_model_batch, batch_size)
        ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
        ref_model_batch = reshuffle_on_epoch_end(ref_model_batch)
        # Generate global reference statistics for diagnostics
        io_ref_stats = ReferenceStatistics(global_ref_models; kwargs_ref_stats...)
        io_ref_models = global_ref_models
    else
        ref_models = construct_reference_models(kwargs_ref_model)
        ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
        io_ref_stats = ref_stats
        io_ref_models = ref_models
        batch_indices = nothing
    end

    outdir_path = create_output_dir(
        ref_stats,
        outdir_root,
        algo_name,
        Δt,
        n_param,
        N_ens,
        N_iter,
        batch_size,
        config_path,
        y_ref_type,
    )

    if !isnothing(prior_μ)
        @assert collect(keys(params)) == collect(keys(prior_μ))
    else
        prior_μ = nothing
    end
    priors = construct_priors(params, outdir_path = outdir_path, unconstrained_σ = unc_σ, prior_mean = prior_μ)
    ekp_kwargs = Dict(:outdir_path => outdir_path, :failure_handler => failure_handler, :localizer => localizer)
    # parameters are sampled in unconstrained space
    if algo_name in ["Inversion", "Sampler", "SparseInversion"]
        if algo_name == "Inversion"
            algo = Inversion()
        elseif algo_name == "SparseInversion"
            γ = get_entry(proc_config, "l1_norm_limit", 1 / eps(Float64))
            prune_below = get_entry(proc_config, "prune_below", 0.0)
            sparse_params = get_entry(proc_config, "sparse_params", true)
            sparse_idx = get_sparse_indices(sparse_params, n_param)
            convex_opt_reg = get_entry(proc_config, "convex_opt_reg", 0.0)
            algo = SparseInversion(γ, prune_below, sparse_idx, convex_opt_reg)
        elseif algo_name == "Sampler"
            algo = Sampler(vcat(mean(priors)...), cov(priors))
        end
        initial_params = construct_initial_ensemble(priors, N_ens, rng_seed = rand(1:1000))
        if augmented
            ekobj = generate_tekp(ref_stats, priors, algo, initial_params; l2_reg = l2_reg, ekp_kwargs...)
        else
            ekobj = generate_ekp(ref_stats, algo, initial_params; ekp_kwargs...)
        end
    elseif algo_name == "Unscented"
        algo = Unscented(vcat(mean(priors)...), cov(priors), α_reg = 1.0, update_freq = 1)
        if augmented
            ekobj = generate_tekp(ref_stats, priors, algo; l2_reg = l2_reg, ekp_kwargs...)
        else
            ekobj = generate_ekp(ref_stats, algo; ekp_kwargs...)
        end
    end

    params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekobj))
    params = [c[:] for c in eachcol(params_cons_i)]
    mod_evaluators = [ModelEvaluator(param, get_name(priors), param_map, ref_models, ref_stats) for param in params]
    versions = generate_scm_input(mod_evaluators, 1, outdir_path, batch_indices)
    # Store version identifiers for this ensemble in a common file
    write_versions(versions, 1, outdir_path = outdir_path)
    # Store ReferenceModelBatch
    if !isnothing(batch_size)
        write_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
    end

    # Initialize validation
    val_ref_models, val_ref_stats =
        isnothing(val_config) ? repeat([nothing], 2) :
        init_validation(val_config, reg_config, namelist_args, ekobj, priors, param_map, versions, outdir_path)

    # Diagnostics IO
    init_diagnostics(config, outdir_path, io_ref_models, io_ref_stats, ekobj, priors, val_ref_models, val_ref_stats)

    if mode == "hpc"
        open("$(job_id).txt", "w") do io
            println(io, outdir_path)
        end
    end
    return outdir_path
end

function get_ref_stats_kwargs(ref_config::Dict{Any, Any}, reg_config::Dict{Any, Any})
    y_ref_type = ref_config["y_reference_type"]
    Σ_ref_type = get_entry(ref_config, "Σ_reference_type", y_ref_type)
    model_errors = get_entry(ref_config, "model_errors", nothing)
    perform_PCA = get_entry(reg_config, "perform_PCA", true)
    variance_loss = get_entry(reg_config, "variance_loss", 1.0e-2)
    normalize = get_entry(reg_config, "normalize", true)
    tikhonov_mode = get_entry(reg_config, "tikhonov_mode", "relative")
    tikhonov_noise = get_entry(reg_config, "tikhonov_noise", 1.0e-6)
    dim_scaling = get_entry(reg_config, "dim_scaling", true)
    return Dict(
        :perform_PCA => perform_PCA,
        :normalize => normalize,
        :variance_loss => variance_loss,
        :tikhonov_noise => tikhonov_noise,
        :tikhonov_mode => tikhonov_mode,
        :dim_scaling => dim_scaling,
        :y_type => y_ref_type,
        :Σ_type => Σ_ref_type,
        :model_errors => model_errors,
    )
end

"Create the calibration output directory and copy the config file into it"
function create_output_dir(
    ref_stats::ReferenceStatistics,
    outdir_root::String,
    algo_name::String,
    Δt::FT,
    n_param::IT,
    N_ens::IT,
    N_iter::IT,
    batch_size::Opt{IT},
    config_path::OptString,
    y_ref_type,
) where {FT <: Real, IT <: Integer}
    # Output path
    d = isnothing(batch_size) ? "d$(pca_length(ref_stats))" : "mb"
    now = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM")
    suffix = randstring(3)  # ensure output folder is unique
    outdir_path = joinpath(
        outdir_root,
        "results_$(algo_name)_dt_$(Δt)_p$(n_param)_e$(N_ens)_i$(N_iter)_$(d)_$(typeof(y_ref_type))_$(now)_$(suffix)",
    )
    @info "Name of outdir path for this EKP is: $outdir_path"
    mkpath(outdir_path)
    if !isnothing(config_path)
        cp(config_path, joinpath(outdir_path, "config.jl"))
    end
    return outdir_path
end

"Initialize the validation process."
function init_validation(
    val_config::Dict{Any, Any},
    reg_config::Dict{Any, Any},
    namelist_args::OptVec{<:Tuple},
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    param_map::ParameterMap,
    versions::Vector{String},
    outdir_path::String;
)

    kwargs_ref_model = get_ref_model_kwargs(val_config; global_namelist_args = namelist_args)
    kwargs_ref_stats = get_ref_stats_kwargs(val_config, reg_config)
    batch_size = get_entry(val_config, "batch_size", nothing)

    if !isnothing(batch_size)
        @info "Validation using mini-batches."
        ref_model_batch = ReferenceModelBatch(kwargs_ref_model)
        ref_models, batch_indices = get_minibatch!(ref_model_batch, batch_size)
        ref_model_batch = reshuffle_on_epoch_end(ref_model_batch)
        write_val_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
    else
        ref_models = construct_reference_models(kwargs_ref_model)
        batch_indices = nothing
    end
    ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
    params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekp))
    params = [c[:] for c in eachcol(params_cons_i)]
    mod_evaluators = [ModelEvaluator(param, get_name(priors), param_map, ref_models, ref_stats) for param in params]
    [
        jldsave(scm_val_init_path(outdir_path, version); model_evaluator, version, batch_indices) for
        (model_evaluator, version) in zip(mod_evaluators, versions)
    ]
    return ref_models, ref_stats
end

"""
    update_validation(
        val_config::Dict{Any, Any},
        reg_config::Dict{Any, Any},
        ekp_old::EnsembleKalmanProcess,
        priors::ParameterDistribution,
        param_map::ParameterMap,
        versions::Vector{String},
        outdir_path::String,
        iteration::Integer
    )

Updates the validation diagnostics and writes to file the validation ModelEvaluators
for the next calibration step.

Inputs:

 - val_config    :: Validation model configuration.
 - reg_config    :: Regularization configuration.
 - ekp_old       :: EnsembleKalmanProcess updated using the past forward model evaluations.
 - priors        :: The priors over parameter space.
 - param_map     :: A mapping to a reduced parameter set. See [`ParameterMap`](@ref) for details.
 - versions      :: String versions identifying the forward model evaluations.
 - outdir_path   :: Output path directory.
 - iteration     :: EKP iteration
"""
function update_validation(
    val_config::Dict{Any, Any},
    reg_config::Dict{Any, Any},
    ekp_old::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    param_map::ParameterMap,
    versions::Vector{String},
    outdir_path::String,
    iteration::Integer,
)

    batch_size = get_entry(val_config, "batch_size", nothing)

    if !isnothing(batch_size)
        ref_model_batch = load(joinpath(outdir_path, "val_ref_model_batch.jld2"))["ref_model_batch"]
        ref_models, batch_indices = get_minibatch!(ref_model_batch, batch_size)
        ref_model_batch = reshuffle_on_epoch_end(ref_model_batch)
        kwargs_ref_stats = get_ref_stats_kwargs(val_config, reg_config)
        ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
        rm(joinpath(outdir_path, "val_ref_model_batch.jld2"))
        write_val_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
    else
        mod_evaluator = load(scm_val_output_path(outdir_path, versions[1]))["model_evaluator"]
        ref_models = mod_evaluator.ref_models
        ref_stats = mod_evaluator.ref_stats
        batch_indices = nothing
    end
    params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekp_old))
    params = [c[:] for c in eachcol(params_cons_i)]
    mod_evaluators = [ModelEvaluator(param, get_name(priors), param_map, ref_models, ref_stats) for param in params]
    # Save new ModelEvaluators using the new versions
    versions = readlines(joinpath(outdir_path, "versions_$(iteration + 1).txt"))
    [
        jldsave(scm_val_init_path(outdir_path, version); model_evaluator, version, batch_indices) for
        (model_evaluator, version) in zip(mod_evaluators, versions)
    ]
    return
end

"""
    ek_update(
        ekobj::EnsembleKalmanProcess,
        priors::ParameterDistribution,
        iteration::Int64,
        config::Dict{Any, Any},
        versions::Vector{String},
        outdir_path::String,
    )

Updates an EnsembleKalmanProcess using forward model evaluations stored
in output files defined by their `versions`, and generates the parameters
for the next ensemble for forward model evaluations. The updated EnsembleKalmanProcess
and new ModelEvaluators are both written to file.

Inputs:

 - ekobj         :: EnsembleKalmanProcess to be updated.
 - priors        :: Priors over parameters, used for unconstrained-constrained mappings.
 - iteration     :: Current iteration of the calibration process.
 - config        :: Process configuration dictionary.
 - versions      :: String versions identifying the forward model evaluations.
 - outdir_path   :: Output path directory.
"""
function ek_update(
    ekobj::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    iteration::Int64,
    config::Dict{Any, Any},
    versions::Vector{String},
    outdir_path::String,
)
    @info "Update the Ensemble Kalman Process at iteration $iteration. $(Dates.now())"
    @info "Output is stored in: $outdir_path"
    # Get config
    proc_config = config["process"]
    N_iter = proc_config["N_iter"]
    algo_name = get_entry(proc_config, "algorithm", "Inversion")

    Δt_scheduler = get_entry(proc_config, "Δt", 1.0)
    Δt = get_Δt(Δt_scheduler, iteration)

    deterministic_forward_map = get_entry(proc_config, "noisy_obs", false)
    augmented = get_entry(proc_config, "augmented", false)
    param_map = get_entry(config["prior"], "param_map", HelperFuncs.do_nothing_param_map())  # do-nothing param map by default

    ref_config = config["reference"]
    batch_size = get_entry(ref_config, "batch_size", nothing)

    val_config = get(config, "validation", nothing)

    reg_config = config["regularization"]
    l2_reg = get_entry(reg_config, "l2_reg", nothing)

    scm_args = load(scm_output_path(outdir_path, versions[1]))
    mod_evaluator = scm_args["model_evaluator"]
    batch_indices = scm_args["batch_indices"]
    ref_stats = mod_evaluator.ref_stats
    ref_models = mod_evaluator.ref_models

    # Advance EKP
    if augmented
        g, g_full = get_ensemble_g_eval_aug(outdir_path, versions, priors, l2_reg)
    else
        g, g_full = get_ensemble_g_eval(outdir_path, versions)
    end

    if isa(ekobj.process, Inversion) || isa(ekobj.process, SparseInversion)
        update_ensemble!(ekobj, g, Δt_new = Δt, deterministic_forward_map = deterministic_forward_map)
    elseif isa(ekobj.process, Unscented)
        update_ensemble!(ekobj, g, Δt_new = Δt)
    else
        Δt ≈ 1.0 ? nothing : @warn "Ensemble Kalman Sampler does not accept a custom Δt."
        update_ensemble!(ekobj, g)
    end

    # Diagnostics IO
    if !isnothing(val_config)
        update_val_diagnostics(outdir_path, ekobj, priors, versions, augmented, l2_reg)
    end
    update_diagnostics(outdir_path, ekobj, priors, ref_stats, g_full, versions, batch_indices)

    if iteration < N_iter
        # Prepare updated EKP and ReferenceModelBatch if minibatching.
        if !isnothing(batch_size)
            ref_model_batch = load(joinpath(outdir_path, "ref_model_batch.jld2"))["ref_model_batch"]
            ekp, ref_models, ref_stats, ref_model_batch, batch_indices =
                update_minibatch_inverse_problem(ref_model_batch, ekobj, priors, batch_size, outdir_path, config)
            rm(joinpath(outdir_path, "ref_model_batch.jld2"))
            write_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
        else
            ekp = ekobj
        end

        # Write to file new EKP and ModelEvaluators
        jldsave(ekobj_path(outdir_path, iteration + 1); ekp)
        write_model_evaluators(ekp, priors, param_map, ref_models, ref_stats, outdir_path, iteration, batch_indices)

        # Update validation ModelEvaluators
        if !isnothing(val_config)
            reg_config = config["regularization"]
            update_validation(val_config, reg_config, ekobj, priors, param_map, versions, outdir_path, iteration)
        end
    end


    # Clean up
    for version in versions
        rm(scm_output_path(outdir_path, version))
        !isnothing(val_config) ? rm(scm_val_output_path(outdir_path, version)) : nothing
    end
    return
end

"""
    restart_calibration(
        ekobj::EnsembleKalmanProcess,
        priors::ParameterDistribution,
        last_iteration::Int64,
        config::Dict{Any, Any},
        outdir_path::String,
    )

Restart a calibration process from an EnsembleKalmanProcess, the parameter priors and the calibration process config file.
If batching, it requires access to the last ReferenceModelBatch, stored in the results directory of the previous 
calibration, `outdir_path`.

Write to file the ModelEvaluators necessary to continue the calibration process.

# Arguments
- `ekobj`           :: EnsembleKalmanProcess to be updated.
- `priors`          :: Priors over parameters, used for unconstrained-constrained mappings.
- `last_iteration`  :: Last iteration of the calibration process to be restarted.
- `config`          :: Configuration dictionary.
- `outdir_path`     :: Output path directory of the calibration process to be restarted.
- `mode`            :: Whether the calibration process is parallelized through HPC resources or using Julia's pmap.
- `job_id`          :: Unique job identifier for sbatch communication.
"""
function restart_calibration(
    ekobj::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    last_iteration::Int64,
    config::Dict{Any, Any},
    outdir_path::String;
    mode::String = "hpc",
    job_id::String = "12345",
)
    @info "Restart calibration on $(Dates.now())"
    # Get config
    ref_config = config["reference"]
    batch_size = get_entry(ref_config, "batch_size", nothing)
    namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    kwargs_ref_model = get_ref_model_kwargs(ref_config; global_namelist_args = namelist_args)

    param_map = get_entry(config["prior"], "param_map", HelperFuncs.do_nothing_param_map())  # do-nothing param map by default

    reg_config = config["regularization"]
    kwargs_ref_stats = get_ref_stats_kwargs(ref_config, reg_config)

    val_config = get(config, "validation", nothing)

    # Prepare updated EKP and ReferenceModelBatch if minibatching.
    if !isnothing(batch_size)
        ref_model_batch = load(joinpath(outdir_path, "ref_model_batch.jld2"))["ref_model_batch"]
        ekp, ref_models, ref_stats, ref_model_batch, batch_indices =
            update_minibatch_inverse_problem(ref_model_batch, ekobj, priors, batch_size, outdir_path, config)
        rm(joinpath(outdir_path, "ref_model_batch.jld2"))
        write_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
    else
        ekp = ekobj
        ref_models = construct_reference_models(kwargs_ref_model)
        ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
        batch_indices = nothing
    end

    # Write to file new EKP and ModelEvaluators
    jldsave(ekobj_path(outdir_path, last_iteration + 1); ekp)
    write_model_evaluators(ekp, priors, param_map, ref_models, ref_stats, outdir_path, last_iteration, batch_indices)

    # Restart validation
    if !isnothing(val_config)
        restart_validation(
            val_config,
            reg_config,
            ekobj,
            priors,
            param_map,
            outdir_path,
            last_iteration,
            namelist_args = namelist_args,
        )
    end
    if mode == "hpc"
        open("$(job_id).txt", "w") do io
            println(io, outdir_path)
        end
    end
    return outdir_path
end

"""
    restart_validation(
        val_config::Dict{Any, Any},
        reg_config::Dict{Any, Any},
        ekp_old::EnsembleKalmanProcess,
        priors::ParameterDistribution,
        param_map::ParameterMap,
        outdir_path::String,
        last_iteration::IT;
        namelist_args = nothing,
    ) where {IT <: Integer}

Restart a validation process by writing to file the validation ModelEvaluators necessary to continue the validation
process. If batching, it requires access to the last validation ReferenceModelBatch, stored in the results directory of
the previous calibration process, `outdir_path`.

# Arguments
- val_config    :: Validation model configuration.
- reg_config    :: Regularization configuration.
- ekp_old       :: EnsembleKalmanProcess updated using the past forward model evaluations.
- priors        :: The priors over parameter space.
- param_map     :: A mapping to a reduced parameter set. See [`ParameterMap`](@ref) for details.
- outdir_path   :: Output path directory.
"""
function restart_validation(
    val_config::Dict{Any, Any},
    reg_config::Dict{Any, Any},
    ekp_old::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    param_map::ParameterMap,
    outdir_path::String,
    last_iteration::IT;
    namelist_args::OptVec{<:Tuple} = nothing,
) where {IT <: Integer}

    kwargs_ref_model = get_ref_model_kwargs(val_config; global_namelist_args = namelist_args)
    kwargs_ref_stats = get_ref_stats_kwargs(val_config, reg_config)
    batch_size = get_entry(val_config, "batch_size", nothing)

    if !isnothing(batch_size)
        ref_model_batch = load(joinpath(outdir_path, "val_ref_model_batch.jld2"))["ref_model_batch"]
        ref_models, batch_indices = get_minibatch!(ref_model_batch, batch_size)
        ref_model_batch = reshuffle_on_epoch_end(ref_model_batch)
        rm(joinpath(outdir_path, "val_ref_model_batch.jld2"))
        write_val_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
    else
        ref_models = construct_reference_models(kwargs_ref_model)
        batch_indices = nothing
    end
    ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)

    params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekp_old))
    params = [c[:] for c in eachcol(params_cons_i)]
    mod_evaluators = [ModelEvaluator(param, get_name(priors), param_map, ref_models, ref_stats) for param in params]
    # Save new ModelEvaluators using the new versions
    versions = readlines(joinpath(outdir_path, "versions_$(last_iteration + 1).txt"))
    [
        jldsave(scm_val_init_path(outdir_path, version); model_evaluator, version, batch_indices) for
        (model_evaluator, version) in zip(mod_evaluators, versions)
    ]
    return
end

"""
    get_ensemble_g_eval(outdir_path::String, versions::Vector{String}; validation::Bool = false)

Recover forward model evaluations from the particle ensemble stored in jld2 files, after which the files are deleted from disk.

# Arguments
- `outdir_path`  :: Path to output directory.
- `versions`     :: Version identifiers of the files containing forward model evaluations.
- `validation`   :: Whether the function should return training or validation forward model evaluations.

# Returns
- `g`            :: Forward model evaluations in the reduced space of the inverse problem.
- `g_full`       :: Forward model evaluations in the original physical space.
"""
function get_ensemble_g_eval(outdir_path::String, versions::Vector{String}; validation::Bool = false)
    # Find train/validation path
    scm_path(x) = validation ? scm_val_output_path(outdir_path, x) : scm_output_path(outdir_path, x)
    # Get array sizes with first file
    scm_outputs = load(scm_path(first(versions)))
    d = length(scm_outputs["g_scm_pca"])
    d_full = length(scm_outputs["g_scm"])
    N_ens = length(versions)
    g = zeros(d, N_ens)
    g_full = zeros(d_full, N_ens)
    for (ens_index, version) in enumerate(versions)
        scm_outputs = load(scm_path(version))
        g[:, ens_index] = scm_outputs["g_scm_pca"]
        g_full[:, ens_index] = scm_outputs["g_scm"]
    end
    return g, g_full
end

"""
    get_ensemble_g_eval_aug(
        outdir_path::String,
        versions::Vector{String},
        priors::ParameterDistribution,
        l2_reg::Union{Dict{String, Vector{R}}, R};
        validation::Bool = false,
    ) where {R}

Recovers forward model evaluations from the particle ensemble stored in jld2 files,
and augments the projected output state with the input parameters in unconstrained form
to enable regularization.

# Arguments
- `outdir_path` :: Path to output directory.
- `versions`    :: Version identifiers of the files containing forward model evaluations.
- `priors`      :: Parameter priors, used to transform between constrained and unconstrained spaces.
- `l2_reg`      :: The config entry specifying l2 regularization.

# Keywords
- `validation`  :: Whether the function should return training or validation forward model evaluations.

# Returns
- `g_aug`   :: Forward model evaluations in the reduced space of the inverse problem, augmented with the unconstrained input parameters.
- `g_full`  :: Forward model evaluations in the original physical space.
"""
function get_ensemble_g_eval_aug(
    outdir_path::String,
    versions::Vector{String},
    priors::ParameterDistribution,
    l2_reg::Union{Dict{String, Vector{R}}, R};
    validation::Bool = false,
) where {R}
    # Find train/validation path
    scm_path(x) = validation ? scm_val_output_path(outdir_path, x) : scm_output_path(outdir_path, x)
    # Get array sizes with first file
    scm_outputs = load(scm_path(first(versions)))

    aug_indices =
        isa(l2_reg, Dict) ? get_regularized_indices(l2_reg) : 1:length(scm_outputs["model_evaluator"].param_cons)

    # Set dimensionality
    d = length(scm_outputs["g_scm_pca"])
    d_aug = d + length(aug_indices)
    d_full = length(scm_outputs["g_scm"])
    N_ens = length(versions)
    g_aug = zeros(d_aug, N_ens)
    g_full = zeros(d_full, N_ens)

    for (ens_index, version) in enumerate(versions)
        scm_outputs = load(scm_path(version))
        g_aug[1:d, ens_index] = scm_outputs["g_scm_pca"]
        g_full[:, ens_index] = scm_outputs["g_scm"]
        θ = transform_constrained_to_unconstrained(priors, scm_outputs["model_evaluator"].param_cons)
        g_aug[(d + 1):d_aug, ens_index] = θ[aug_indices]
    end
    return g_aug, g_full
end

"""
   versioned_model_eval(version, outdir_path, mode, config)

Perform or omit a model evaluation given the parsed mode and provided config, and write to file the model output.

# Arguments
- version       :: The version associated with the ModelEvaluator to be used.
- outdir_path   :: The path to the results directory of the calibration process.
- mode          :: Whether the ModelEvaluator is used for training or validation.
- config        :: The general configuration dictionary.
"""
function versioned_model_eval(version::String, outdir_path::String, mode::String, config::Dict)
    @assert mode in ["train", "validation"]
    # Omits validation if unsolicited
    if mode == "validation" && isnothing(get(config, "validation", nothing))
        return
    elseif mode == "validation"
        input_path = scm_val_init_path(outdir_path, version)
        output_path = scm_val_output_path(outdir_path, version)
    else
        input_path = scm_init_path(outdir_path, version)
        output_path = scm_output_path(outdir_path, version)
    end
    # Load inputs
    scm_args = load(input_path)
    failure_handler = get_entry(config["process"], "failure_handler", "high_loss")
    # Check consistent failure method for given algorithm
    @assert failure_handler == "sample_succ_gauss" ? config["process"]["algorithm"] != "Sampler" : true
    model_evaluator = scm_args["model_evaluator"]
    batch_indices = scm_args["batch_indices"]
    # Eval
    sim_dirs, g_scm, g_scm_pca = run_SCM(model_evaluator, failure_handler = failure_handler)
    # Store output and delete input
    jldsave(output_path; sim_dirs, g_scm, g_scm_pca, model_evaluator, version, batch_indices)

    # Save full output timeseries to `<results_folder>/timeseries/<version>/Output.<case>.<case_id>`
    save_tc_output = get(config["output"], "save_tc_output", false)
    if save_tc_output
        save_tc_data(config, outdir_path, sim_dirs, version, mode)
    end

    rm(input_path)
end

"""
    update_minibatch_inverse_problem(
        rm_batch::ReferenceModelBatch,
        ekp_old::EnsembleKalmanProcess,
        priors::ParameterDistribution,
        batch_size::Integer,
        outdir_path::String,
        config::Dict{Any, Any},
    )

Return the [`EnsembleKalmanProcess`](@ref) and [`ReferenceStatistics`](@ref) consistent with the new [`ReferenceModel`](@ref)
minibatch, and update the evaluation order of the [`ReferenceModelBatch`](@ref).

# Arguments
- `rm_batch`    :: The global `ReferenceModelBatch` with the current model evaluation order.
- `ekp_old`     :: The `EnsembleKalmanProcess` from the previous minibatch evaluation.
- `batch_size`  :: The batch size of the current minibatch.
- `outdir_path` :: The output directory.
- `config`      :: The configuration dictionary.

# Returns
- `ekp`             :: The `EnsembleKalmanProcess` for the current minibatch.
- `ref_models`      :: The current minibatch of `ReferenceModel`s.
- `ref_stats`       :: The `ReferenceStatistics` consistent with the current minibatch.
- `ref_model_batch` :: The global `ReferenceModelBatch` with the updated model evaluation order.
"""
function update_minibatch_inverse_problem(
    rm_batch::ReferenceModelBatch,
    ekp_old::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    batch_size::Integer,
    outdir_path::String,
    config::Dict{Any, Any},
)
    # Construct new reference minibatch, new ref_stats, and new ekp
    ref_model_batch = deepcopy(rm_batch)
    ref_models, batch_indices = get_minibatch!(ref_model_batch, batch_size)
    ref_model_batch = reshuffle_on_epoch_end(ref_model_batch)

    ref_config = config["reference"]
    reg_config = config["regularization"]
    proc_config = config["process"]

    augmented = get_entry(proc_config, "augmented", false)
    failure_handler = get_entry(proc_config, "failure_handler", "high_loss")
    localizer = get_entry(proc_config, "localizer", NoLocalization())
    l2_reg = get_entry(reg_config, "l2_reg", nothing)
    kwargs_ref_stats = get_ref_stats_kwargs(ref_config, reg_config)
    ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
    process = ekp_old.process

    ekp_kwargs = Dict(:outdir_path => outdir_path, :failure_handler => failure_handler, :localizer => localizer)

    if isa(process, Unscented)
        # Reconstruct UKI using regularization toward the prior
        algo = Unscented(
            process.u_mean[end],
            process.uu_cov[end],
            α_reg = 1.0,
            update_freq = 1,
            prior_mean = vcat(mean(priors)...),
        )
        if augmented
            ekp = generate_tekp(ref_stats, priors, algo; l2_reg = l2_reg, ekp_kwargs...)
        else
            ekp = generate_ekp(ref_stats, algo; ekp_kwargs...)
        end
    else
        if augmented
            ekp = generate_tekp(ref_stats, priors, process, get_u_final(ekp_old); l2_reg = l2_reg, ekp_kwargs...)
        else
            ekp = generate_ekp(ref_stats, process, get_u_final(ekp_old); ekp_kwargs...)
        end
    end
    return ekp, ref_models, ref_stats, ref_model_batch, batch_indices
end

"""
    write_model_evaluators(
        ekp::EnsembleKalmanProcess,
        priors::ParameterDistribution,
        param_map::ParameterMap,
        ref_models::Vector{ReferenceModel},
        ref_stats::ReferenceStatistics,
        outdir_path::String,
        iteration::Int,
    )

Create and write to file the [`ModelEvaluator`](@ref)s for the current particle ensemble.

# Arguments
- `ekp`         :: The `EnsembleKalmanProcess` with the current ensemble of parameter values.
- `priors`      :: The parameter priors.
- `param_map`   :: A mapping to a reduced parameter set. See [`ParameterMap`](@ref) for details.
- `ref_models`  :: The `ReferenceModel`s defining the new model evaluations.
- `ref_stats`   :: The `ReferenceStatistics` corresponding to passed `ref_models`.
- `outdir_path` :: The output directory.
- `iteration`   :: The current process iteration.
"""
function write_model_evaluators(
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    param_map::ParameterMap,
    ref_models::Vector{ReferenceModel},
    ref_stats::ReferenceStatistics,
    outdir_path::String,
    iteration::Int,
    batch_indices::OptVec{Int} = nothing,
)
    params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekp))
    params = [c[:] for c in eachcol(params_cons_i)]
    mod_evaluators = [ModelEvaluator(param, get_name(priors), param_map, ref_models, ref_stats) for param in params]
    versions = generate_scm_input(mod_evaluators, iteration + 1, outdir_path, batch_indices)
    # Store version identifiers for this ensemble in a common file
    write_versions(versions, iteration + 1, outdir_path = outdir_path)
    return
end

"""
    init_diagnostics(
        config::Dict{Any, Any},
        outdir_path::String,
        ref_stats::ReferenceStatistics,
        ekp::EnsembleKalmanProcess,
        priors::ParameterDistribution,
    )

Create a diagnostics netcdf file.

# Arguments
- `config`      :: User-defined configuration dictionary.
- `outdir_path` :: Path of results directory.
- `ref_stats`   :: `ReferenceStatistics`.
- `ekp`         :: Initial `EnsembleKalmanProcess`, containing parameter information, but no forward model evaluations.
- `priors`      :: Prior distributions of the parameters.
"""
function init_diagnostics(
    config::Dict{Any, Any},
    outdir_path::String,
    ref_models::Vector{ReferenceModel},
    ref_stats::ReferenceStatistics,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    val_ref_models::OptVec{ReferenceModel} = nothing,
    val_ref_stats::Opt{ReferenceStatistics} = nothing,
)
    write_full_stats = get_entry(config["reference"], "write_full_stats", true)
    N_ens = size(get_u_final(ekp), 2)
    diags = NetCDFIO_Diags(config, outdir_path, ref_stats, N_ens, priors, val_ref_stats)
    # Write prior and reference diagnostics
    io_prior(diags, priors)
    io_reference(diags, ref_stats, ref_models, write_full_stats)
    # Add diags, write first state diags
    init_iteration_io(diags)
    init_metrics(diags)
    init_ensemble_diags(diags, ekp, priors)
    init_particle_diags(diags, ekp, priors)
    if !isnothing(val_ref_models)
        write_full_stats = get_entry(config["validation"], "write_full_stats", true)
        init_val_diagnostics(diags, val_ref_stats, val_ref_models, write_full_stats)
    end
end

"""
    update_diagnostics(outdir_path::String, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Append diagnostics of the current iteration evaluations (i.e., forward model output metrics) and the next iteration state 
(i.e., parameters and parameter metrics) to a diagnostics netcdf file.

# Arguments
- `outdir_path`     :: Path of results directory.
- `ekp`             :: Current `EnsembleKalmanProcess`.
- `priors`          :: Prior distributions of the parameters.
- `ref_stats`       :: `ReferenceStatistics`.
- `g_full`          :: The forward model evaluation in primitive space.
- `versions`        :: Version identifiers of the forward model evaluations at the current iteration.
- `val_config`      :: The validation configuration, if given.
- `batch_indices`   :: The indices of the `ReferenceModel`s used in the current batch.
"""
function update_diagnostics(
    outdir_path::String,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    ref_stats::ReferenceStatistics,
    g_full::Array{<:Real, 2},
    versions::Union{Vector{Int}, Vector{String}},
    batch_indices::OptVec{Int} = nothing,
)

    mse_full = compute_mse(g_full, ref_stats.y_full)
    diags = NetCDFIO_Diags(joinpath(outdir_path, "Diagnostics.nc"))

    io_diagnostics(diags, ekp, priors, mse_full, g_full, batch_indices)
end

function update_val_diagnostics(
    outdir_path::String,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    versions::Union{Vector{Int}, Vector{String}},
    augmented::Bool,
    l2_reg,
)
    scm_args = load(scm_val_output_path(outdir_path, versions[1]))
    mod_evaluator = scm_args["model_evaluator"]
    val_batch_indices = scm_args["batch_indices"]
    val_ref_stats = mod_evaluator.ref_stats
    if augmented
        g, g_full = get_ensemble_g_eval_aug(outdir_path, versions, priors, l2_reg, validation = true)
    else
        g, g_full = get_ensemble_g_eval(outdir_path, versions, validation = true)
    end
    # Compute diagnostics
    mse_full = compute_mse(g_full, val_ref_stats.y_full)
    diags = NetCDFIO_Diags(joinpath(outdir_path, "Diagnostics.nc"))
    io_val_diagnostics(diags, ekp, mse_full, g, g_full, val_ref_stats, val_batch_indices)
end

end # module
