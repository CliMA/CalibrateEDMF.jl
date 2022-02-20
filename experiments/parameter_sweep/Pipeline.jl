module ParameterSweepPipeline

using Random
using JLD2

using CalibrateEDMF
using CalibrateEDMF.Pipeline
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.NetCDFIO
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

export init_sweep, versioned_model_eval

"""
    init_sweep(config::Dict{Any, Any}, job_id::String, , config_path)

Initializes a calibration process given a configuration, and a pipeline mode.

    Inputs:
    - config :: User-defined configuration dictionary.
    - job_id :: Unique job identifier for sbatch communication.
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

    constraints = config["prior"]["constraints"]
    # unc_σ = get_entry(config["prior"], "unconstrained_σ", 1.0)
    # prior_μ_dict = get_entry(config["prior"], "prior_mean", nothing)

    namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    # Dimensionality
    n_param = sum(map(length, collect(values(constraints))))

    ref_models = construct_reference_models(kwargs_ref_model)
    # Create input scm stats and namelist file if files don't already exist
    run_reference_SCM.(ref_models, overwrite = overwrite_scm_file, namelist_args = namelist_args)
    ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)
    io_ref_stats = ref_stats
    io_ref_models = ref_models

    outdir_path = create_sweep_output_dir(
        ref_stats,
        outdir_root,
        n_param,
        N_ens²,
        batch_size,
        config_path,
        y_ref_type,
    )

    priors = construct_priors(constraints, outdir_path = outdir_path)
    constraints_keys =  collect(keys(constraints))
    lower_bounds_1 = constraints[constraints_keys[1]][1].constrained_to_unconstrained.lower_bound
    lower_bounds_2 = constraints[constraints_keys[2]][1].constrained_to_unconstrained.lower_bound
    upper_bounds_1 = constraints[constraints_keys[1]][1].constrained_to_unconstrained.upper_bound
    upper_bounds_2 = constraints[constraints_keys[2]][1].constrained_to_unconstrained.upper_bound
    vals_param_1 = LinRange(lower_bounds_1, upper_bounds_1, N_ens)
    vals_param_2 = LinRange(lower_bounds_2, upper_bounds_2, N_ens)
    params = vec(collect(Iterators.product(vals_param_1, vals_param_2)))
    ϕ = hcat([collect(elem) for elem in params]...)
    mod_evaluators = [ModelEvaluator([param...], get_name(priors), ref_models, ref_stats) for param in params]
    versions = generate_scm_input(mod_evaluators, outdir_path)
    # Store version identifiers for this ensemble in a common file
    write_versions(versions, 1, outdir_path = outdir_path)
    init_sweep_diagnostics()
        config,
        outdir_path,
        io_ref_models,
        io_ref_stats,
        N_ens²,
        ϕ,
        priors,
    )

    if mode == "hpc"
        open("$(job_id).txt", "w") do io
            write(io, "$(outdir_path)\n")
        end
    elseif mode == "pmap"
        return outdir_path
    end
end

# function get_ref_model_kwargs(ref_config::Dict{Any, Any})
#     n_cases = length(ref_config["case_name"])
#     Σ_dir = expand_dict_entry(ref_config, "Σ_dir", n_cases)
#     Σ_t_start = expand_dict_entry(ref_config, "Σ_t_start", n_cases)
#     Σ_t_end = expand_dict_entry(ref_config, "Σ_t_end", n_cases)
#     return Dict(
#         :y_names => ref_config["y_names"],
#         # Reference path specification
#         :y_dir => ref_config["y_dir"],
#         :Σ_dir => Σ_dir,
#         :scm_parent_dir => ref_config["scm_parent_dir"],
#         :scm_suffix => ref_config["scm_suffix"],
#         # Case name
#         :case_name => ref_config["case_name"],
#         # Define observation window (s)
#         :t_start => ref_config["t_start"],
#         :t_end => ref_config["t_end"],
#         :Σ_t_start => Σ_t_start,
#         :Σ_t_end => Σ_t_end,
#     )
# end

# function get_ref_stats_kwargs(ref_config::Dict{Any, Any}, reg_config::Dict{Any, Any})
#     y_ref_type = ref_config["y_reference_type"]
#     Σ_ref_type = get_entry(ref_config, "Σ_reference_type", y_ref_type)
#     perform_PCA = get_entry(reg_config, "perform_PCA", true)
#     variance_loss = get_entry(reg_config, "variance_loss", 1.0e-2)
#     normalize = get_entry(reg_config, "normalize", true)
#     tikhonov_mode = get_entry(reg_config, "tikhonov_mode", "relative")
#     tikhonov_noise = get_entry(reg_config, "tikhonov_noise", 1.0e-6)
#     dim_scaling = get_entry(reg_config, "dim_scaling", true)
#     return Dict(
#         :perform_PCA => perform_PCA,
#         :normalize => normalize,
#         :variance_loss => variance_loss,
#         :tikhonov_noise => tikhonov_noise,
#         :tikhonov_mode => tikhonov_mode,
#         :dim_scaling => dim_scaling,
#         :y_type => y_ref_type,
#         :Σ_type => Σ_ref_type,
#     )
# end


"Create the calibration output directory and copy the config file into it"
function create_sweep_output_dir(
    ref_stats::ReferenceStatistics,
    outdir_root::String,
    n_param::IT,
    N_ens::IT,
    batch_size::Union{IT, Nothing},
    config_path::Union{String, Nothing},
    y_ref_type,
) where {FT <: Real, IT <: Integer}
    # Output path
    d = isnothing(batch_size) ? "d$(pca_length(ref_stats))" : "mb"
    outdir_path = joinpath(
        outdir_root,
        "results_sweep_p$(n_param)_e$(N_ens)_$(d)_$(typeof(y_ref_type))_$(rand(11111:99999))",
    )
    @info "Name of outdir path for this EKP is: $outdir_path"
    mkpath(outdir_path)
    if !isnothing(config_path)
        cp(config_path, joinpath(outdir_path, "config.jl"))
    end
    return outdir_path
end

# """
#     get_ensemble_g_eval(outdir_path::String, versions::Vector{String})

# Recovers forward model evaluations from the particle ensemble stored in jld2 files,
# after which the files are deleted from disk.

# Inputs:
#  - outdir_path  :: Path to output directory.
#  - versions     :: Version identifiers of the files containing forward model evaluations.
# Outputs:
#  - g            :: Forward model evaluations in the reduced space of the inverse problem.
#  - g_full       :: Forward model evaluations in the original physical space.
# """
# function get_ensemble_g_eval(outdir_path::String, versions::Vector{String}; validation::Bool = false)
#     # Find train/validation path
#     @info("start get_ensemble_g_eval")
#     scm_path(x) = validation ? scm_val_output_path(outdir_path, x) : scm_output_path(outdir_path, x)
#     # Get array sizes with first file
#     scm_outputs = load(scm_path(versions[1]))
#     d = length(scm_outputs["g_scm_pca"])
#     d_full = length(scm_outputs["g_scm"])
#     N_ens = length(versions)
#     g = zeros(d, N_ens)
#     g_full = zeros(d_full, N_ens)
#     for (ens_index, version) in enumerate(versions)
#         scm_outputs = load(scm_path(version))
#         g[:, ens_index] = scm_outputs["g_scm_pca"]
#         g_full[:, ens_index] = scm_outputs["g_scm"]
#     end
#     @info("finished get_ensemble_g_eval")
#     return g, g_full
# end

"""
   versioned_model_eval(version::Union{String, Int}, outdir_path::String, config::Dict{Any, Any})

Performs or omits a model evaluation given the parsed mode and provided config,
 and writes to file the model output.

Inputs:
 - version       :: The version associated with the ModelEvaluator to be used.
 - outdir_path   :: The path to the results directory of the calibration process.
 - config        :: The general configuration dictionary.
"""
function versioned_model_eval(version::Union{String, Int}, outdir_path::String, config::Dict{Any, Any})
    # Omits validation if unsolicited
    input_path = scm_init_path(outdir_path, version)
    output_path = scm_output_path(outdir_path, version)
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
 - ϕ           :: The parameter values.
 - priors      :: The parameter priors.
 - ref_models  :: The ReferenceModels defining the new model evaluations.
 - ref_stats   :: The ReferenceStatistics corresponding to passed ref_models.
 - outdir_path :: The output directory.
 - iteration   :: The current process iteration.
"""
function write_model_evaluators(
    ϕ::Matrix{FT},
    priors::ParameterDistribution,
    ref_models::Vector{ReferenceModel},
    ref_stats::ReferenceStatistics,
    outdir_path::String,
    iteration::Int,
) where{FT}
    @info("start write_model_evaluators")
    params = [c[:] for c in eachcol(ϕ)]
    mod_evaluators = [ModelEvaluator(param, get_name(priors), ref_models, ref_stats) for param in params]
    versions = generate_scm_input(mod_evaluators, outdir_path)
    # Store version identifiers for this ensemble in a common file
    write_versions(versions, iteration + 1, outdir_path = outdir_path)
    @info("finish write_model_evaluators")
    return
end

"""
    init_sweep_diagnostics(
        config::Dict{Any, Any},
        outdir_path::String,
        ref_stats::ReferenceStatistics,
        N_ens::number of ensemble members,
        ϕ:: parameter values,
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
function init_sweep_diagnostics(
    config::Dict{Any, Any},
    outdir_path::String,
    ref_models::Vector{ReferenceModel},
    ref_stats::ReferenceStatistics,
    N_ens::Int64,
    ϕ::Matrix{FT},
    priors::ParameterDistribution,
    #val_ref_models::Union{Vector{ReferenceModel}, Nothing},
    #val_ref_stats::Union{ReferenceStatistics, Nothing},
) where{FT}
    write_full_stats = get_entry(config["reference"], "write_full_stats", true)
    diags = NetCDFIO_Diags(config, outdir_path, ref_stats, N_ens, priors, nothing)
    # Write reference
    # io_reference(diags, ref_stats, ref_models, write_full_stats)
    # Add diags, write first state diags
    init_iteration_io(diags)
    # init_metrics(diags)
    # init_ensemble_diags(diags, ϕ, priors)
    init_particle_diags(diags, ϕ, priors)
end

"""
    update_sweep_diagnostics(outdir_path::String, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Appends diagnostics of the current iteration evaluations (i.e., forward model output metrics)
and the next iteration state (i.e., parameters and parameter metrics) to a diagnostics netcdf file.

    Inputs:
    - outdir_path :: Path of results directory.
    - priors:: Prior distributions of the parameters.
    - ref_stats :: ReferenceStatistics.
    - g_full :: The forward model evaluation in primitive space.
    - versions :: Version identifiers of the forward model evaluations at the current iteration.
"""
function update_sweep_diagnostics(
    outdir_path::String,
    priors::ParameterDistribution,
    ref_stats::ReferenceStatistics,
    g_full::Array{FT, 2},
    versions::Union{Vector{Int}, Vector{String}},
) where {FT <: Real}

    @info("start update_sweep_diagnostics")
    mse_full = compute_mse(g_full, ref_stats.y_full)
    diags = NetCDFIO_Diags(joinpath(outdir_path, "Diagnostics.nc"))
    @info("finish update_sweep_diagnostics")
end

"""
    write_sweep_diagnostics(
        priors::ParameterDistribution,
        versions::Vector{String},
        outdir_path::String,
    )

Stored in output files defined by their `versions`
Inputs:

 - priors        :: Priors over parameters, used for unconstrained-constrained mappings.
 - versions      :: String versions identifying the forward model evaluations.
 - outdir_path   :: Output path directory.
"""
function write_sweep_diagnostics(
    priors::ParameterDistribution,
    versions::Vector{String},
    outdir_path::String,
)
    @info("start write_sweep_diagnostics")
    mod_evaluator = load(scm_output_path(outdir_path, versions[1]))["model_evaluator"]
    ref_stats = mod_evaluator.ref_stats
    g, g_full = get_ensemble_g_eval(outdir_path, versions)
    update_sweep_diagnostics(outdir_path, priors, ref_stats, g_full, versions)
    @info("finish write_sweep_diagnostics")
    return
end

end # module
