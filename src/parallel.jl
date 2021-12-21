
using Distributed
using CalibrateEDMF
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
using TurbulenceConvection
include(joinpath(@__DIR__, "helper_funcs.jl"))

export run_SCM_parallel, eval_single_ref_model, versioned_model_eval_parallel

function run_SCM_parallel(
    ME::ModelEvaluator; 
    error_check::Bool = false, namelist_args = nothing,
    particle_failure_fixer = "high_loss",
    ) where {FT <: Real}
    return run_SCM_parallel(
        ME.param_cons,
        ME.param_names,
        ME.ref_models,
        ME.ref_stats,
        error_check = error_check,
        namelist_args = namelist_args,
        particle_failure_fixer = particle_failure_fixer,
    )
end

function run_SCM_parallel(
    u::Vector{FT},
    u_names::Vector{String},
    RM::Vector{ReferenceModel},
    RS::ReferenceStatistics;
    error_check::Bool = false,
    namelist_args = nothing,
    particle_failure_fixer = "high_loss",
) where {FT <: Real}

    mkpath(joinpath(pwd(), "tmp"))
    result_arr = pmap(x -> eval_single_ref_model(x..., RS, u, u_names, namelist_args), enumerate(RM))
    # Unpack
    sim_dirs, g_scm, g_scm_pca, sim_errors = (getindex.(result_arr, i) for i in 1:4)
    g_scm = vcat(g_scm...)
    g_scm_pca = vcat(g_scm_pca...)
    model_error = any(sim_errors)

    # penalize nan-values in output
    any(isnan.(g_scm)) && @warn("NaN-values in output data")
    if particle_failure_fixer == "cond_success_update"
        nothing
    elseif particle_failure_fixer == "high_loss"
        g_scm[isnan.(g_scm)] .= 1e5
        g_scm_pca[isnan.(g_scm_pca)] .= 1e5
    else
        @warn("No known particle failure handler used")
    end
    @info "Length of g_scm (full): $(length(g_scm))"
    @info "Length of g_scm (pca) : $(length(g_scm_pca))"
    if error_check
        return sim_dirs, g_scm, g_scm_pca, model_error
    else
        return sim_dirs, g_scm, g_scm_pca
    end
end

function eval_single_ref_model(
    m_index::IT,
    m::ReferenceModel,
    RS::ReferenceStatistics,
    u::Vector{FT},
    u_names::Vector{String},
    namelist_args = nothing,
) where {FT <: Real, IT <: Int}
    # create temporary directory to store SCM data in
    tmpdir = mktempdir(joinpath(pwd(), "tmp"))
    # run TurbulenceConvection.jl. Get output directory for simulation data
    sim_dir, model_error = run_SCM_handler(m, tmpdir, u, u_names, namelist_args)
    g_scm = get_profile(m, sim_dir, z_scm = get_height(sim_dir))
    g_scm = normalize_profile(g_scm, length(m.y_names), RS.norm_vec[m_index])
    # perform PCA reduction
    g_scm_pca = RS.pca_vec[m_index]' * g_scm
    return sim_dir, g_scm, g_scm_pca, model_error
end

function versioned_model_eval_parallel(
    version::Union{String, Int},
    outdir_path::String,
    mode::String,
    config::Dict{Any, Any},
)
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
    particle_failure_fixer = get_entry(config["process"], "particle_failure_fixer", "high_loss")

    sim_dirs, g_scm, g_scm_pca = run_SCM_parallel(
        model_evaluator, namelist_args = namelist_args,
        particle_failure_fixer = particle_failure_fixer,
    )
    # Store output and delete input
    jldsave(output_path; sim_dirs, g_scm, g_scm_pca, model_evaluator, version)
    rm(input_path)
end
