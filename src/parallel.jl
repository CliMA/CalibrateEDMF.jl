
using Distributed
using CalibrateEDMF
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
using TurbulenceConvection
using CalibrateEDMF.HelperFuncs
import CalibrateEDMF.TurbulenceConvectionUtils: eval_single_ref_model

export run_SCM_parallel, versioned_model_eval_parallel

function run_SCM_parallel(ME::ModelEvaluator; error_check::Bool = false, failure_handler = "high_loss")
    return run_SCM_parallel(
        ME.param_cons,
        ME.param_names,
        ME.param_map,
        ME.ref_models,
        ME.ref_stats,
        error_check = error_check,
        failure_handler = failure_handler,
    )
end

function run_SCM_parallel(
    u::Vector{FT},
    u_names::Vector{String},
    param_map::ParameterMap,
    RM::Vector{ReferenceModel},
    RS::ReferenceStatistics;
    error_check::Bool = false,
    failure_handler = "high_loss",
) where {FT <: Real}

    mkpath(joinpath(pwd(), "tmp"))
    result_arr = pmap(x -> eval_single_ref_model(x..., RS, u, u_names, param_map), enumerate(RM))
    # Unpack
    sim_dirs, g_scm, g_scm_pca, sim_errors = (getindex.(result_arr, i) for i in 1:4)
    g_scm = vcat(g_scm...)
    g_scm_pca = vcat(g_scm_pca...)
    model_error = any(sim_errors)

    # penalize nan-values in output
    if model_error && failure_handler == "high_loss"
        g_scm .= 1e5
        g_scm_pca .= 1e5
    end

    @info "Length of g_scm (full, pca): ($(length(g_scm)), $(length(g_scm_pca)))"
    if error_check
        return sim_dirs, g_scm, g_scm_pca, model_error
    else
        return sim_dirs, g_scm, g_scm_pca
    end
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
    model_evaluator = scm_args["model_evaluator"]
    batch_indices = scm_args["batch_indices"]
    # Eval
    failure_handler = get_entry(config["process"], "failure_handler", "high_loss")

    sim_dirs, g_scm, g_scm_pca = run_SCM_parallel(model_evaluator, failure_handler = failure_handler)

    # Store output and delete input
    jldsave(output_path; sim_dirs, g_scm, g_scm_pca, model_evaluator, version, batch_indices)
    rm(input_path)
end
