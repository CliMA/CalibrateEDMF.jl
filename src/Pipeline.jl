module Pipeline

using Random
using JLD2

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

export init_calibration, ek_update

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

    ref_config = config["reference"]
    n_cases = length(ref_config["case_name"])
    y_ref_type = ref_config["y_reference_type"]
    Σ_ref_type = get_entry(ref_config, "Σ_reference_type", y_ref_type)
    Σ_dir = expand_dict_entry(ref_config, "Σ_dir", n_cases)
    Σ_t_start = expand_dict_entry(ref_config, "Σ_t_start", n_cases)
    Σ_t_end = expand_dict_entry(ref_config, "Σ_t_end", n_cases)
    batch_size = get_entry(ref_config, "batch_size", nothing)

    reg_config = config["regularization"]
    perform_PCA = get_entry(reg_config, "perform_PCA", true)
    variance_loss = get_entry(reg_config, "variance_loss", 1.0e-2)
    normalize = get_entry(reg_config, "normalize", true)
    tikhonov_mode = get_entry(reg_config, "tikhonov_mode", "relative")
    tikhonov_noise = get_entry(reg_config, "tikhonov_noise", 1.0e-6)
    dim_scaling = get_entry(reg_config, "dim_scaling", true)

    out_config = config["output"]
    save_eki_data = get_entry(out_config, "save_eki_data", true)
    save_ensemble_data = get_entry(out_config, "save_ensemble_data", false)
    overwrite_scm_file = get_entry(out_config, "overwrite_scm_file", false)
    outdir_root = get_entry(out_config, "outdir_root", pwd())

    proc_config = config["process"]
    N_ens = proc_config["N_ens"]
    N_iter = proc_config["N_iter"]
    algo_name = get_entry(proc_config, "algorithm", "Inversion")
    Δt = get_entry(proc_config, "Δt", 1.0)

    params = config["prior"]["constraints"]
    unc_σ = get_entry(config["prior"], "unconstrained_σ", 1.0)

    namelist_args = get_entry(config["scm"], "namelist_args", nothing)

    # Dimensionality
    n_param = sum(map(length, collect(values(params))))
    if algo_name == "Unscented"
        N_ens = 2 * n_param + 1
        @warn "Number of ensemble members overwritten to 2p + 1 for Unscented Kalman Inversion."
    end

    # Construct reference models
    kwargs_ref_model = Dict(
        :y_names => ref_config["y_names"],
        # Reference path specification
        :y_dir => ref_config["y_dir"],
        :Σ_dir => Σ_dir,
        :scm_parent_dir => ref_config["scm_parent_dir"],
        :scm_suffix => ref_config["scm_suffix"],
        # Case name
        :case_name => ref_config["case_name"],
        # Define observation window (s)
        :t_start => ref_config["t_start"],
        :t_end => ref_config["t_end"],
        :Σ_t_start => Σ_t_start,
        :Σ_t_end => Σ_t_end,
    )

    # Minibatch mode
    if !isnothing(batch_size)
        @info "Training using mini-batches."
        ref_model_batch = construct_ref_model_batch(kwargs_ref_model)
        global_ref_models = deepcopy(ref_model_batch.ref_models)
        # Create input scm stats and namelist file if files don't already exist
        run_reference_SCM.(global_ref_models, overwrite = overwrite_scm_file, namelist_args = namelist_args)
        # Generate global reference statistics
        global_ref_stats = ReferenceStatistics(
            global_ref_models,
            perform_PCA,
            normalize,
            variance_loss = variance_loss,
            tikhonov_noise = tikhonov_noise,
            tikhonov_mode = tikhonov_mode,
            dim_scaling = dim_scaling,
            y_type = y_ref_type,
            Σ_type = Σ_ref_type,
        )
        ref_models = get_minibatch!(ref_model_batch, batch_size)
    else
        ref_models = construct_reference_models(kwargs_ref_model)
        # Create input scm stats and namelist file if files don't already exist
        run_reference_SCM.(ref_models, overwrite = overwrite_scm_file, namelist_args = namelist_args)
    end
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

    # Output path
    d = isnothing(batch_size) ? "d$(length(ref_stats.y))" : "mb"
    outdir_path = joinpath(
        outdir_root,
        "results_$(algo_name)_dt$(Δt)_p$(n_param)_e$(N_ens)_i$(N_iter)_$(d)_$(typeof(y_ref_type))_$(rand(11111:99999))",
    )
    @info "Name of outdir path for this EKP is: $outdir_path"
    mkpath(outdir_path)
    if !isnothing(config_path)
        cp(config_path, joinpath(outdir_path, "config.jl"))
    end

    priors = construct_priors(params, outdir_path = outdir_path, unconstrained_σ = unc_σ)
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
    if !isnothing(batch_size)
        init_diagnostics(config, outdir_path, global_ref_stats, global_ref_models, ekobj, priors)
    else
        init_diagnostics(config, outdir_path, ref_stats, ref_models, ekobj, priors)
    end

    if mode == "hpc"
        open("$(job_id).txt", "w") do io
            write(io, "$(outdir_path)\n")
        end
        params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekobj))
        params = [c[:] for c in eachcol(params_cons_i)]
        mod_evaluators = [ModelEvaluator(param, get_name(priors), ref_models, ref_stats) for param in params]
        versions = generate_scm_input(mod_evaluators, outdir_path)
        # Store version identifiers for this ensemble in a common file
        write_versions(versions, 1, outdir_path = outdir_path)
        # Store ReferenceModelBatch
        if !isnothing(batch_size)
            write_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
        end
    elseif mode == "pmap"
        if !isnothing(batch_size)
            return Dict(
                "ekobj" => ekobj,
                "priors" => priors,
                "ref_stats" => ref_stats,
                "ref_models" => ref_models,
                "ref_model_batch" => ref_model_batch,
                "outdir_path" => outdir_path,
            )
        else
            return Dict(
                "ekobj" => ekobj,
                "priors" => priors,
                "ref_stats" => ref_stats,
                "ref_models" => ref_models,
                "outdir_path" => outdir_path,
            )
        end
    end
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
    # Get dimensionality
    proc_config = config["process"]
    N_iter = proc_config["N_iter"]
    algo_name = get_entry(proc_config, "algorithm", "Inversion")
    Δt = get_entry(proc_config, "Δt", 1.0)
    deterministic_forward_map = get_entry(proc_config, "noisy_obs", false)

    ref_config = config["reference"]
    batch_size = get_entry(ref_config, "batch_size", nothing)

    mod_evaluator = load(scm_output_path(outdir_path, versions[1]))["model_evaluator"]
    ref_stats = mod_evaluator.ref_stats
    ref_models = mod_evaluator.ref_models

    # Advance EKP
    g, g_full = get_ensemble_g_eval(outdir_path, versions)
    # Scale artificial timestep by batch size
    Δt_scaled = Δt / length(ref_models)
    if isa(ekobj.process, Inversion)
        update_ensemble!(ekobj, g, Δt_new = Δt_scaled, deterministic_forward_map = deterministic_forward_map)
    else
        update_ensemble!(ekobj, g)
    end
    # Diagnostics IO
    update_diagnostics(outdir_path, ekobj, priors, ref_stats, g_full, batch_size)

    if !isnothing(batch_size)
        ref_model_batch = load(joinpath(outdir_path, "ref_model_batch.jld2"))["ref_model_batch"]
        ekp, ref_models, ref_stats, ref_model_batch =
            update_minibatch_inverse_problem(ref_model_batch, ekobj, batch_size, outdir_path, config)
        rm(joinpath(outdir_path, "ref_model_batch.jld2"))
        write_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
    else
        ekp = ekobj
    end
    jldsave(ekobj_path(outdir_path, iteration + 1); ekp)
    write_model_evaluators(ekp, priors, ref_models, ref_stats, outdir_path, iteration)
    return
end

"""
    get_ensemble_g_eval(outdir_path::String, versions::Vector{String})

Recovers forward model evaluations from the particle ensemble stored in jld2 files,
after which the files are deleted from disk.

Inputs:
 - outdir_path  :: Path to output directory.
 - versions     :: Version identifiers of the files containing forward model evaluations.
Outputs:
 - g            :: Forward model evaluations in the reduced space of the inverse problem.
 - g_full       :: Forward model evaluations in the original physical space.
"""
function get_ensemble_g_eval(outdir_path::String, versions::Vector{String})
    # Get array sizes with first file
    scm_outputs = load(scm_output_path(outdir_path, versions[1]))
    d = length(scm_outputs["g_scm_pca"])
    d_full = length(scm_outputs["g_scm"])
    N_ens = length(versions)
    g = zeros(d, N_ens)
    g_full = zeros(d_full, N_ens)
    for (ens_index, version) in enumerate(versions)
        scm_outputs = load(scm_output_path(outdir_path, version))
        g[:, ens_index] = scm_outputs["g_scm_pca"]
        g_full[:, ens_index] = scm_outputs["g_scm"]
        # Clean up
        rm(scm_output_path(outdir_path, version))
        rm(scm_init_path(outdir_path, version))
    end
    return g, g_full
end

"""
    update_minibatch_inverse_problem(
        rm_batch::ReferenceModelBatch,
        ekp_old::EnsembleKalmanProcess,
        batch_size::Integer,
        outdir_path::String,
        config::Dict{Any, Any},
    )

Returns the EnsembleKalmanProcess and ReferenceStatistics consistent with the
new ReferenceModel minibatch, and updates the evaluation order of the ReferenceModelBatch.

Inputs:
 - rm_batch    :: The global ReferenceModelBatch with the current model evaluation order.
 - ekp_old     :: The EnsembleKalmanProcess from the previous minibatch evaluation.
 - batch_size  :: The batch size of the current minibatch.
 - outdir_path :: The output directory.
 - config      :: The configuration dictionary.
Outputs:
 - ekp             :: The EnsembleKalmanProcess for the current minibatch.
 - ref_models      :: The current minibatch of ReferenceModels.
 - ref_stats       :: The ReferenceStatistics consistent with the current minibatch.
 - ref_model_batch :: The global ReferenceModelBatch with the updated model evaluation order.
"""
function update_minibatch_inverse_problem(
    rm_batch::ReferenceModelBatch,
    ekp_old::EnsembleKalmanProcess,
    batch_size::Integer,
    outdir_path::String,
    config::Dict{Any, Any},
)
    # Construct new reference minibatch, new ref_stats, and new ekp
    ref_model_batch = deepcopy(rm_batch)
    ref_models = get_minibatch!(ref_model_batch, batch_size)
    if isempty(ref_model_batch.eval_order)
        @info "Current epoch finished. Reshuffling dataset."
        ref_model_batch = construct_ref_model_batch(ref_model_batch.ref_models)
    else
        ref_model_batch = ref_model_batch
    end

    ref_config = config["reference"]
    y_ref_type = ref_config["y_reference_type"]
    Σ_ref_type = get_entry(ref_config, "Σ_reference_type", y_ref_type)

    reg_config = config["regularization"]
    perform_PCA = get_entry(reg_config, "perform_PCA", true)
    variance_loss = get_entry(reg_config, "variance_loss", 1.0e-2)
    normalize = get_entry(reg_config, "normalize", true)
    tikhonov_mode = get_entry(reg_config, "tikhonov_mode", "relative")
    tikhonov_noise = get_entry(reg_config, "tikhonov_noise", 1.0e-6)
    dim_scaling = get_entry(reg_config, "dim_scaling", true)

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
    if isa(ekp_old.process, Inversion) || isa(ekp_old.process, Sampler)
        ekp = generate_ekp(get_u_final(ekp_old), ref_stats, ekp_old.process, outdir_path = outdir_path)
    elseif isa(ekp_old.process, Unscented)
        # α == 1.0 to have a consistent reconstructed Unscented Kalman Process
        algo = Unscented(ekp_old.process.u_mean[end], ekp_old.process.uu_cov[end], 1.0, 1)
        ekp = generate_ekp(ref_stats, algo, outdir_path = outdir_path)
    else
        throw(ArgumentError("Process must be an Inversion, Sampler or Unscented Kalman process."))
    end
    return ekp, ref_models, ref_stats, ref_model_batch
end

"""
    write_model_evaluators(
        ekp::EnsembleKalmanProcess,
        priors::ParameterDistribution,
        ref_models::Vector{ReferenceModel},
        ref_stats::ReferenceStatistics,
        outdir_path::String,
        iteration::Int,
    )

Creates and writes to file the ModelEvaluators for the current particle ensemble.

Inputs:
 - ekp         :: The EnsembleKalmanProcess with the current ensemble of parameter values.
 - priors      :: The parameter priors.
 - ref_models  :: The ReferenceModels defining the new model evaluations.
 - ref_stats   :: The ReferenceStatistics corresponding to passed ref_models.
 - outdir_path :: The output directory.
 - iteration   :: The current process iteration.
"""
function write_model_evaluators(
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    ref_models::Vector{ReferenceModel},
    ref_stats::ReferenceStatistics,
    outdir_path::String,
    iteration::Int,
)
    params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekp))
    params = [c[:] for c in eachcol(params_cons_i)]
    mod_evaluators = [ModelEvaluator(param, get_name(priors), ref_models, ref_stats) for param in params]
    versions = generate_scm_input(mod_evaluators, outdir_path)
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
    diags = NetCDFIO_Diags(config, outdir_path, ref_stats, ekp, priors)
    # Write reference
    io_reference(diags, ref_stats, ref_models)
    # Add metric fields
    init_metrics(diags)
    # Add diags, write first state diags
    init_iteration_io(diags)
    init_ensemble_diags(diags, ekp, priors)
    init_particle_diags(diags, ekp, priors)
end

"""
    update_diagnostics(outdir_path::String, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Appends current iteration diagnostics to a diagnostics netcdf file.

    Inputs:
    - outdir_path :: Path of results directory.
    - ekp :: Current EnsembleKalmanProcess.
    - priors:: Prior distributions of the parameters.
    - ref_stats :: ReferenceStatistics.
    - g_full :: The forward model evaluation in primitive space.
"""
function update_diagnostics(
    outdir_path::String,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    ref_stats::ReferenceStatistics,
    g_full::Array{FT, 2},
    batch_size::Union{Int, Nothing},
) where {FT <: Real}
    # Compute diagnostics
    mse_full = compute_mse(g_full, ref_stats.y_full)
    diags = NetCDFIO_Diags(joinpath(outdir_path, "Diagnostics.nc"))
    if isnothing(batch_size)
        io_diagnostics(diags, ekp, priors, mse_full, g_full)
    else
        io_diagnostics(diags, ekp, priors, mse_full)
    end
end

end # module
