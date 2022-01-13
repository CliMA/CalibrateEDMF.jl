module ParameterSweepPipeline

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
using EnsembleKalmanProcesses.ParameterDistributionStorage

export init_sweep, versioned_model_eval

"""
    init_calibration(job_id::String, config::Dict{Any, Any})

Initializes a calibration process given a configuration, and a pipeline mode.

    Inputs:
    - job_id :: Unique job identifier for sbatch communication.
    - config :: User-defined configuration dictionary.
    - mode :: Whether the calibration process is parallelized through HPC resources
      or using Julia's pmap.
"""
function init_sweep(config::Dict{Any, Any}; mode::String = "hpc", job_id::String = "12345", config_path = nothing)
    @assert mode in ["hpc", "pmap"]

    ref_config = config["reference"]
    y_ref_type = ref_config["y_reference_type"]
    batch_size = get_entry(ref_config, "batch_size", nothing)
    kwargs_ref_model = get_ref_model_kwargs(ref_config)

    reg_config = config["regularization"]
    kwargs_ref_stats = get_ref_stats_kwargs(ref_config, reg_config)

    out_config = config["output"]
    save_eki_data = get_entry(out_config, "save_eki_data", true)
    save_ensemble_data = get_entry(out_config, "save_ensemble_data", false)
    overwrite_scm_file = get_entry(out_config, "overwrite_scm_file", false)
    outdir_root = get_entry(out_config, "outdir_root", pwd())

    proc_config = config["process"]
    N_ens = proc_config["N_ens"]
    N_ens² = N_ens*N_ens
    # N_iter = proc_config["N_iter"]
    # algo_name = get_entry(proc_config, "algorithm", "Inversion")
    # Δt = get_entry(proc_config, "Δt", 1.0)

    params = config["prior"]["constraints"]
    # unc_σ = get_entry(config["prior"], "unconstrained_σ", 1.0)
    # prior_μ_dict = get_entry(config["prior"], "prior_mean", nothing)

    namelist_args = get_entry(config["scm"], "namelist_args", nothing)

    # val_config = get(config, "validation", nothing)

    # Dimensionality
    n_param = sum(map(length, collect(values(params))))

    ref_models = construct_reference_models(kwargs_ref_model)
    # Create input scm stats and namelist file if files don't already exist
    run_reference_SCM.(ref_models, overwrite = overwrite_scm_file, namelist_args = namelist_args)
    ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
    io_ref_stats = ref_stats
    io_ref_models = ref_models

    outdir_path = create_output_dir(
        ref_stats,
        outdir_root,
        n_param,
        N_ens²,
        batch_size,
        config_path,
        y_ref_type,
    )

    priors = construct_priors(params, outdir_path = outdir_path)
    # parameters are sampled in unconstrained space

    params_cons_i = get_value(N_ens²,params, constrainst) # get the 100x2 sets of parameter values with reperitions
    # @warn size(params_cons_i)
    # println(size(params_cons_i))
    params = [c[:] for c in eachcol(params_cons_i)] # ?? make sure the shape of params_cons_i is correct
    mod_evaluators = [ModelEvaluator(param, get_name(priors), ref_models, ref_stats) for param in params]
    versions = generate_scm_input(mod_evaluators, outdir_path)
    # Store version identifiers for this ensemble in a common file
    write_versions(versions, 1, outdir_path = outdir_path)

    # Diagnostics IO - I need to create a new init_diagnostics that not require ekobj, val_ref_models, val_ref_stats
    # new SweepDiagnostic.jl in src
    init_diagnostics(
        config,
        outdir_path,
        io_ref_models,
        io_ref_stats,
        ekobj,
        priors,
        val_ref_models,
        val_ref_stats,
        !isnothing(val_config),
    )

    if mode == "hpc"
        open("$(job_id).txt", "w") do io
            write(io, "$(outdir_path)\n")
        end
    elseif mode == "pmap"
        return outdir_path
    end
end

function get_ref_model_kwargs(ref_config::Dict{Any, Any})
    n_cases = length(ref_config["case_name"])
    Σ_dir = expand_dict_entry(ref_config, "Σ_dir", n_cases)
    Σ_t_start = expand_dict_entry(ref_config, "Σ_t_start", n_cases)
    Σ_t_end = expand_dict_entry(ref_config, "Σ_t_end", n_cases)
    return Dict(
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
end

function get_ref_stats_kwargs(ref_config::Dict{Any, Any}, reg_config::Dict{Any, Any})
    y_ref_type = ref_config["y_reference_type"]
    Σ_ref_type = get_entry(ref_config, "Σ_reference_type", y_ref_type)
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
    batch_size::Union{IT, Nothing},
    config_path::Union{String, Nothing},
    y_ref_type,
) where {FT <: Real, IT <: Integer}
    # Output path
    d = isnothing(batch_size) ? "d$(pca_length(ref_stats))" : "mb"
    outdir_path = joinpath(
        outdir_root,
        "results_$(algo_name)_dt$(Δt)_p$(n_param)_e$(N_ens)_i$(N_iter)_$(d)_$(typeof(y_ref_type))_$(rand(11111:99999))",
    )
    @info "Name of outdir path for this EKP is: $outdir_path"
    mkpath(outdir_path)
    if !isnothing(config_path)
        cp(config_path, joinpath(outdir_path, "config.jl"))
    end
    return outdir_path
end

# "Initialize the validation process."
# function init_validation(
#     val_config::Dict{Any, Any},
#     reg_config::Dict{Any, Any},
#     ekp::EnsembleKalmanProcess,
#     priors::ParameterDistribution,
#     versions::Vector{IT},
#     outdir_path::String;
#     overwrite::Bool = false,
#     namelist_args = nothing,
# ) where {FT <: Real, IT <: Integer}

#     kwargs_ref_model = get_ref_model_kwargs(val_config)
#     kwargs_ref_stats = get_ref_stats_kwargs(val_config, reg_config)
#     batch_size = get_entry(val_config, "batch_size", nothing)

#     if !isnothing(batch_size)
#         @info "Validation using mini-batches."
#         ref_model_batch = construct_ref_model_batch(kwargs_ref_model)
#         run_reference_SCM.(ref_model_batch.ref_models, overwrite = overwrite, namelist_args = namelist_args)
#         ref_models = get_minibatch!(ref_model_batch, batch_size)
#         ref_model_batch = reshuffle_on_epoch_end(ref_model_batch)
#         write_val_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
#     else
#         ref_models = construct_reference_models(kwargs_ref_model)
#         run_reference_SCM.(ref_models, overwrite = overwrite, namelist_args = namelist_args)
#     end
#     ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
#     params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekp))
#     params = [c[:] for c in eachcol(params_cons_i)]
#     mod_evaluators = [ModelEvaluator(param, get_name(priors), ref_models, ref_stats) for param in params]
#     [
#         jldsave(scm_val_init_path(outdir_path, version); model_evaluator, version)
#         for (model_evaluator, version) in zip(mod_evaluators, versions)
#     ]
#     return ref_models, ref_stats
# end

# """
#     update_validation(
#         val_config::Dict{Any, Any},
#         reg_config::Dict{Any, Any},
#         ekp_old::EnsembleKalmanProcess,
#         priors::ParameterDistribution,
#         versions::Vector{String},
#         outdir_path::String,
#         iteration::IT
#         )

# Updates the validation diagnostics and writes to file the validation ModelEvaluators
# for the next calibration step.

# Inputs:

#  - val_config    :: Validation model configuration.
#  - reg_config    :: Regularization configuration.
#  - ekp_old       :: EnsembleKalmanProcess updated using the past forward model evaluations.
#  - priors        :: The priors over parameter space.
#  - versions      :: String versions identifying the forward model evaluations.
#  - outdir_path   :: Output path directory.
# """
# function update_validation(
#     val_config::Dict{Any, Any},
#     reg_config::Dict{Any, Any},
#     ekp_old::EnsembleKalmanProcess,
#     priors::ParameterDistribution,
#     versions::Vector{String},
#     outdir_path::String,
#     iteration::IT,
# ) where {IT <: Integer}

#     batch_size = get_entry(val_config, "batch_size", nothing)

#     if !isnothing(batch_size)
#         ref_model_batch = load(joinpath(outdir_path, "val_ref_model_batch.jld2"))["ref_model_batch"]
#         ref_models = get_minibatch!(ref_model_batch, batch_size)
#         ref_model_batch = reshuffle_on_epoch_end(ref_model_batch)
#         kwargs_ref_stats = get_ref_stats_kwargs(val_config, reg_config)
#         ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
#         rm(joinpath(outdir_path, "val_ref_model_batch.jld2"))
#         write_val_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
#     else
#         mod_evaluator = load(scm_val_output_path(outdir_path, versions[1]))["model_evaluator"]
#         ref_models = mod_evaluator.ref_models
#         ref_stats = mod_evaluator.ref_stats
#     end
#     params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekp_old))
#     params = [c[:] for c in eachcol(params_cons_i)]
#     mod_evaluators = [ModelEvaluator(param, get_name(priors), ref_models, ref_stats) for param in params]
#     # Save new ModelEvaluators using the new versions
#     versions = readlines(joinpath(outdir_path, "versions_$(iteration + 1).txt"))
#     [
#         jldsave(scm_val_init_path(outdir_path, version); model_evaluator, version)
#         for (model_evaluator, version) in zip(mod_evaluators, versions)
#     ]
#     return
# end

# """
#     ek_update(
#         ekobj::EnsembleKalmanProcess,
#         priors::ParameterDistribution,
#         iteration::Int64,
#         config::Dict{Any, Any},
#         versions::Vector{String},
#         outdir_path::String,
#     )

# Updates an EnsembleKalmanProcess using forward model evaluations stored
# in output files defined by their `versions`, and generates the parameters
# for the next ensemble for forward model evaluations. The updated EnsembleKalmanProcess
# and new ModelEvaluators are both written to file.

# Inputs:

#  - ekobj         :: EnsembleKalmanProcess to be updated.
#  - priors        :: Priors over parameters, used for unconstrained-constrained mappings.
#  - iteration     :: Current iteration of the calibration process.
#  - config        :: Process configuration dictionary.
#  - versions      :: String versions identifying the forward model evaluations.
#  - outdir_path   :: Output path directory.
# """
# function ek_update(
#     ekobj::EnsembleKalmanProcess,
#     priors::ParameterDistribution,
#     iteration::Int64,
#     config::Dict{Any, Any},
#     versions::Vector{String},
#     outdir_path::String,
# )
#     # Get config
#     proc_config = config["process"]
#     N_iter = proc_config["N_iter"]
#     algo_name = get_entry(proc_config, "algorithm", "Inversion")
#     Δt = get_entry(proc_config, "Δt", 1.0)
#     deterministic_forward_map = get_entry(proc_config, "noisy_obs", false)

#     ref_config = config["reference"]
#     batch_size = get_entry(ref_config, "batch_size", nothing)

#     val_config = get(config, "validation", nothing)

#     mod_evaluator = load(scm_output_path(outdir_path, versions[1]))["model_evaluator"]
#     ref_stats = mod_evaluator.ref_stats
#     ref_models = mod_evaluator.ref_models

#     # Advance EKP
#     g, g_full = get_ensemble_g_eval(outdir_path, versions)
#     # Scale artificial timestep by batch size
#     Δt_scaled = Δt / length(ref_models)
#     if isa(ekobj.process, Inversion)
#         update_ensemble!(ekobj, g, Δt_new = Δt_scaled, deterministic_forward_map = deterministic_forward_map)
#     else
#         update_ensemble!(ekobj, g)
#     end

#     # Diagnostics IO
#     update_diagnostics(outdir_path, ekobj, priors, ref_stats, g_full, batch_size, versions, val_config)

#     if iteration < N_iter
#         # Prepare updated EKP and ReferenceModelBatch if minibatching.
#         if !isnothing(batch_size)
#             ref_model_batch = load(joinpath(outdir_path, "ref_model_batch.jld2"))["ref_model_batch"]
#             ekp, ref_models, ref_stats, ref_model_batch =
#                 update_minibatch_inverse_problem(ref_model_batch, ekobj, batch_size, outdir_path, config)
#             rm(joinpath(outdir_path, "ref_model_batch.jld2"))
#             write_ref_model_batch(ref_model_batch, outdir_path = outdir_path)
#         else
#             ekp = ekobj
#         end

#         # Write to file new EKP and ModelEvaluators
#         jldsave(ekobj_path(outdir_path, iteration + 1); ekp)
#         write_model_evaluators(ekp, priors, ref_models, ref_stats, outdir_path, iteration)

#         # Update validation ModelEvaluators
#         if !isnothing(val_config)
#             reg_config = config["regularization"]
#             update_validation(val_config, reg_config, ekobj, priors, versions, outdir_path, iteration)
#         end
#     end
#     # Clean up
#     for version in versions
#         rm(scm_output_path(outdir_path, version))
#         !isnothing(val_config) ? rm(scm_val_output_path(outdir_path, version)) : nothing
#     end
#     return
# end

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
function get_ensemble_g_eval(outdir_path::String, versions::Vector{String}; validation::Bool = false)
    # Find train/validation path
    scm_path(x) = validation ? scm_val_output_path(outdir_path, x) : scm_output_path(outdir_path, x)
    # Get array sizes with first file
    scm_outputs = load(scm_path(versions[1]))
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
   versioned_model_eval(version::Union{String, Int}, outdir_path::String, mode::String, config::Dict{Any, Any})

Performs or omits a model evaluation given the parsed mode and provided config,
 and writes to file the model output.

Inputs:
 - version       :: The version associated with the ModelEvaluator to be used.
 - outdir_path   :: The path to the results directory of the calibration process.
 - mode          :: Whether the ModelEvaluator is used for training or validation.
 - config        :: The general configuration dictionary.
"""
function versioned_model_eval(version::Union{String, Int}, outdir_path::String, mode::String, config::Dict{Any, Any})
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
    namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    model_evaluator = scm_args["model_evaluator"]
    # Eval
    sim_dirs, g_scm, g_scm_pca = run_SCM(model_evaluator, namelist_args = namelist_args)
    # Store output and delete input
    jldsave(output_path; sim_dirs, g_scm, g_scm_pca, model_evaluator, version)
    rm(input_path)
end

# """
#     update_minibatch_inverse_problem(
#         rm_batch::ReferenceModelBatch,
#         ekp_old::EnsembleKalmanProcess,
#         batch_size::Integer,
#         outdir_path::String,
#         config::Dict{Any, Any},
#     )

# Returns the EnsembleKalmanProcess and ReferenceStatistics consistent with the
# new ReferenceModel minibatch, and updates the evaluation order of the ReferenceModelBatch.

# Inputs:
#  - rm_batch    :: The global ReferenceModelBatch with the current model evaluation order.
#  - ekp_old     :: The EnsembleKalmanProcess from the previous minibatch evaluation.
#  - batch_size  :: The batch size of the current minibatch.
#  - outdir_path :: The output directory.
#  - config      :: The configuration dictionary.
# Outputs:
#  - ekp             :: The EnsembleKalmanProcess for the current minibatch.
#  - ref_models      :: The current minibatch of ReferenceModels.
#  - ref_stats       :: The ReferenceStatistics consistent with the current minibatch.
#  - ref_model_batch :: The global ReferenceModelBatch with the updated model evaluation order.
# """
# function update_minibatch_inverse_problem(
#     rm_batch::ReferenceModelBatch,
#     ekp_old::EnsembleKalmanProcess,
#     batch_size::Integer,
#     outdir_path::String,
#     config::Dict{Any, Any},
# )
#     # Construct new reference minibatch, new ref_stats, and new ekp
#     ref_model_batch = deepcopy(rm_batch)
#     ref_models = get_minibatch!(ref_model_batch, batch_size)
#     ref_model_batch = reshuffle_on_epoch_end(ref_model_batch)

#     ref_config = config["reference"]
#     reg_config = config["regularization"]
#     kwargs_ref_stats = get_ref_stats_kwargs(ref_config, reg_config)

#     ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
#     if isa(ekp_old.process, Inversion) || isa(ekp_old.process, Sampler)
#         ekp = generate_ekp(get_u_final(ekp_old), ref_stats, ekp_old.process, outdir_path = outdir_path)
#     elseif isa(ekp_old.process, Unscented)
#         # α == 1.0 to have a consistent reconstructed Unscented Kalman Process
#         algo = Unscented(ekp_old.process.u_mean[end], ekp_old.process.uu_cov[end], 1.0, 1)
#         ekp = generate_ekp(ref_stats, algo, outdir_path = outdir_path)
#     else
#         throw(ArgumentError("Process must be an Inversion, Sampler or Unscented Kalman process."))
#     end
#     return ekp, ref_models, ref_stats, ref_model_batch
# end

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
    ref_models::Vector{ReferenceModel},
    ref_stats::ReferenceStatistics,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    val_ref_models::Union{Vector{ReferenceModel}, Nothing},
    val_ref_stats::Union{ReferenceStatistics, Nothing},
    validation::Bool = false,
)
    write_full_stats = get_entry(config["reference"], "write_full_stats", true)
    diags = NetCDFIO_Diags(config, outdir_path, ref_stats, ekp, priors, val_ref_stats)
    # Write reference
    io_reference(diags, ref_stats, ref_models, write_full_stats)
    # Add diags, write first state diags
    init_iteration_io(diags)
    init_metrics(diags)
    init_ensemble_diags(diags, ekp, priors)
    init_particle_diags(diags, ekp, priors)
    if validation
        write_full_stats = get_entry(config["validation"], "write_full_stats", true)
        init_val_diagnostics(diags, val_ref_stats, val_ref_models, write_full_stats)
    end
end

"""
    update_diagnostics(outdir_path::String, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Appends diagnostics of the current iteration evaluations (i.e., forward model output metrics)
and the next iteration state (i.e., parameters and parameter metrics) to a diagnostics netcdf file.

    Inputs:
    - outdir_path :: Path of results directory.
    - ekp :: Current EnsembleKalmanProcess.
    - priors:: Prior distributions of the parameters.
    - ref_stats :: ReferenceStatistics.
    - g_full :: The forward model evaluation in primitive space.
    - batch_size :: The number of evaluations per minibatch, if minibatching.
    - versions :: Version identifiers of the forward model evaluations at the current iteration.
    - val_config :: The validation configuration, if given.
"""
function update_diagnostics(
    outdir_path::String,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    ref_stats::ReferenceStatistics,
    g_full::Array{FT, 2},
    batch_size::Union{Int, Nothing},
    versions::Union{Vector{Int}, Vector{String}},
    val_config::Union{Dict{Any, Any}, Nothing} = nothing,
) where {FT <: Real}

    if !isnothing(val_config)
        update_val_diagnostics(outdir_path, versions, val_config)
    end
    mse_full = compute_mse(g_full, ref_stats.y_full)
    diags = NetCDFIO_Diags(joinpath(outdir_path, "Diagnostics.nc"))
    if isnothing(batch_size)
        io_diagnostics(diags, ekp, priors, mse_full, g_full)
    else
        io_diagnostics(diags, ekp, priors, mse_full)
    end
end

# function update_val_diagnostics(
#     outdir_path::String,
#     versions::Union{Vector{Int}, Vector{String}},
#     val_config::Dict{Any, Any},
# ) where {FT <: Real}
#     batch_size = get_entry(val_config, "batch_size", nothing)
#     mod_evaluator = load(scm_val_output_path(outdir_path, versions[1]))["model_evaluator"]
#     ref_stats = mod_evaluator.ref_stats
#     g, g_full = get_ensemble_g_eval(outdir_path, versions, validation = true)
#     # Compute diagnostics
#     mse_full = compute_mse(g_full, ref_stats.y_full)
#     diags = NetCDFIO_Diags(joinpath(outdir_path, "Diagnostics.nc"))
#     if isnothing(batch_size)
#         io_val_diagnostics(diags, mse_full, g, g_full)
#     else
#         io_val_diagnostics(diags, mse_full)
#     end
# end

end # module