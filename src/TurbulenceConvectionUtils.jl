module TurbulenceConvectionUtils

import Logging
using JLD2
using JSON
using Random
using DocStringExtensions

using ..ReferenceModels
import ..ReferenceModels: NameList

using ..ReferenceStats
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributions
import EnsembleKalmanProcesses: construct_initial_ensemble
using TurbulenceConvection

using ..HelperFuncs

export ModelEvaluator
export run_SCM, run_SCM_handler, run_reference_SCM
export generate_scm_input, get_gcm_les_uuid, eval_single_ref_model
export save_full_ensemble_data
export precondition

"""
    ModelEvaluator

A structure containing the information required to perform
a forward model evaluation at a given parameter vector.

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
Base.@kwdef struct ModelEvaluator{FT <: Real}
    "Parameter vector in constrained (i.e. physical) space"
    param_cons::Vector{FT}
    "Parameter names associated with parameter vector"
    param_names::Vector{String}
    "Vector of reference models"
    ref_models::Vector{ReferenceModel}
    "Reference statistics for the inverse problem"
    ref_stats::ReferenceStatistics

    function ModelEvaluator(
        param_cons::Vector{FT},
        param_names::Vector{String},
        RM::Vector{ReferenceModel},
        RS::ReferenceStatistics,
    ) where {FT <: Real}
        return new{FT}(param_cons, param_names, RM, RS)
    end
end


"""
    run_SCM(
        u::Vector{FT},
        u_names::Vector{String},
        RM::Vector{ReferenceModel},
        RS::ReferenceStatistics;
        error_check::Bool = false,
        namelist_args = nothing,
        failure_handler = "high_loss",
    ) where {FT <: Real}
    run_SCM(
        ME::ModelEvaluator;
        error_check::Bool = false,
        namelist_args = nothing,
        failure_handler = "high_loss",
    ) where {FT <: Real}

Run the single-column model (SCM) using a set of parameters
and return the value of outputs defined in y_names, possibly
after normalization and projection onto lower dimensional
space using PCA.

The function also outputs a boolean diagnosing whether the
requested SCM simulation failed.

Inputs:

 - `u`               :: Values of parameters to be used in simulations.
 - `u_names`         :: SCM names for parameters `u`.
 - `RM`              :: Vector of `ReferenceModel`s
 - `RS`              :: reference statistics for simulation
 - `error_check`     :: Returns as an additional argument whether the SCM call errored.
 - `namelist_args`   :: Additional arguments passed to the TurbulenceConvection namelist.
 - `failure_handler` :: Method used to handle failed simulations.

Outputs:

 - `sim_dirs`    :: Vector of simulation output directories
 - `g_scm`       :: Vector of model evaluations concatenated for all flow configurations.
 - `g_scm_pca`   :: Projection of `g_scm` onto principal subspace spanned by eigenvectors.
 - `model_error` :: Whether the simulation errored with the requested configuration.
"""
function run_SCM(
    u::Vector{FT},
    u_names::Vector{String},
    RM::Vector{ReferenceModel},
    RS::ReferenceStatistics;
    error_check::Bool = false,
    namelist_args = nothing,
    failure_handler = "high_loss",
) where {FT <: Real}

    mkpath(joinpath(pwd(), "tmp"))
    result_arr = map(x -> eval_single_ref_model(x..., RS, u, u_names, namelist_args), enumerate(RM))
    # Unpack
    sim_dirs, g_scm, g_scm_pca, sim_errors = (getindex.(result_arr, i) for i in 1:4)
    g_scm = vcat(g_scm...)
    g_scm_pca = vcat(g_scm_pca...)

    model_error = any(sim_errors)
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
function run_SCM(
    ME::ModelEvaluator;
    error_check::Bool = false,
    namelist_args = nothing,
    failure_handler = "high_loss",
) where {FT <: Real}
    return run_SCM(
        ME.param_cons,
        ME.param_names,
        ME.ref_models,
        ME.ref_stats,
        error_check = error_check,
        namelist_args = namelist_args,
        failure_handler = failure_handler,
    )
end

"""
    eval_single_ref_model(
        m_index::IT,
        m::ReferenceModel,
        RS::ReferenceStatistics,
        u::Vector{FT},
        u_names::Vector{String},
        namelist_args = nothing,
    ) where {FT <: Real, IT <: Int}

Runs the single-column model (SCM) under a single configuration
(i.e., ReferenceModel) using a set of parameters u, and returns
the forward model evaluation in both the original and the latent
PCA space.

Inputs:

 - `m_index`       :: The index of the ReferenceModel within the overarching
                    ref_models vector used to construct the ReferenceStatistics.
 - `m`             :: A ReferenceModel.
 - `RS`            :: reference statistics for simulation
 - `u`             :: Values of parameters to be used in simulations.
 - `u_names`       :: SCM names for parameters `u`.
 - `namelist_args` :: Additional arguments passed to the TurbulenceConvection namelist.

Outputs:

 - `sim_dir`     ::  Simulation output directory.
 - `g_scm`       :: Forward model evaluation in original output space.
 - `g_scm_pca`   :: Projection of `g_scm` onto principal subspace spanned by eigenvectors.
 - `model_error` :: Whether the simulation errored with the requested configuration.
"""
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
    filename = get_stats_path(sim_dir)
    z_obs = get_z_obs(m)
    if model_error
        g_scm = get_profile(m, filename, z_scm = z_obs) # Get shape
        g_scm = fill(NaN, length(g_scm))
    else
        g_scm, prof_indices = get_profile(m, filename, z_scm = z_obs, prof_ind = true)
        g_scm = normalize_profile(g_scm, RS.norm_vec[m_index], length(z_obs), prof_indices)
    end
    # perform PCA reduction
    g_scm_pca = RS.pca_vec[m_index]' * g_scm
    return sim_dir, g_scm, g_scm_pca, model_error
end

"""
    run_reference_SCM(m::ReferenceModel; overwrite::Bool = false, run_single_timestep = true)

Run the single-column model (SCM) for a reference model object
using default parameters, and write the output to file.

Inputs:
 - `m`                    :: A `ReferenceModel`.
 - `overwrite`            :: if true, run TC.jl and overwrite existing simulation files.
 - `run_single_timestep`  :: if true, run only one time step.
"""
function run_reference_SCM(m::ReferenceModel; overwrite::Bool = false, run_single_timestep = true)
    output_dir = scm_dir(m)
    if ~isdir(joinpath(output_dir, "stats")) | overwrite
        namelist = get_scm_namelist(m)

        default_t_max = namelist["time_stepping"]["t_max"]
        default_adapt_dt = namelist["time_stepping"]["adapt_dt"]
        if run_single_timestep
            # Run only 1 timestep -- since we don't need output data, only simulation config
            namelist["time_stepping"]["adapt_dt"] = false
            namelist["time_stepping"]["t_max"] = namelist["time_stepping"]["dt_min"]
        end
        # run TurbulenceConvection.jl
        logger = Logging.ConsoleLogger(stderr, Logging.Warn)
        _, ret_code = Logging.with_logger(logger) do
            main1d(namelist; time_run = false)
        end
        if ret_code ≠ :success
            @warn "Default TurbulenceConvection.jl simulation $(basename(m.y_dir)) failed."
        end
        if run_single_timestep
            # reset t_max to default and overwrite stored namelist file
            namelist["time_stepping"]["t_max"] = default_t_max
            namelist["time_stepping"]["adapt_dt"] = default_adapt_dt
            open(namelist_directory(output_dir, m), "w") do io
                JSON.print(io, namelist, 4)
            end
        end
    end
end


"""
    run_SCM_handler(
        m::ReferenceModel,
        tmpdir::String,
        u::Array{FT, 1},
        u_names::Array{String, 1},
        namelist_args = nothing,
    ) where {FT<:AbstractFloat}

Run a case using a set of parameters `u_names` with values `u`,
and return directory pointing to where data is stored for simulation run.

Inputs:

 - m             :: Reference model
 - tmpdir        :: Temporary directory to store simulation results in
 - u             :: Values of parameters to be used in simulations.
 - u_names       :: SCM names for parameters `u`.
 - namelist_args :: Additional arguments passed to the TurbulenceConvection namelist.

Outputs:

 - output_dir   :: directory containing output data from the SCM run.
 - model_error   :: Boolean specifying whether the simulation failed.
"""
function run_SCM_handler(
    m::ReferenceModel,
    tmpdir::String,
    u::Vector{FT},
    u_names::Vector{String},
    namelist_args = nothing,
) where {FT <: AbstractFloat}

    # fetch namelist
    namelist = get_scm_namelist(m)
    # output subset of variables needed for calibration
    namelist["stats_io"]["calibrate_io"] = true

    # run TurbulenceConvection.jl with modified parameters
    return run_SCM_handler(
        m.case_name,
        tmpdir;
        u = u,
        u_names = u_names,
        namelist = namelist,
        namelist_args = namelist_args,
        uuid = basename(tmpdir), # set random uuid
        les = get(namelist["meta"], "lesfile", nothing),
    )
end


"""
    run_SCM_handler(
        case_name::String,
        out_dir::String;
        u::Vector{FT},
        u_names::Vector{String},
        namelist::Union{Dict, Nothing} = nothing,
        namelist_args::Union{Tuple, Nothing} = nothing,
        uuid::String = "01",
        les::Union{NamedTuple, String} = nothing,
    )

Run a TurbulenceConvection.jl case and return directory pointing to where data is stored for simulation run.

Inputs:
 - case_name     :: case name
 - out_dir       :: Directory to store simulation results in.
 Optional Inputs:
 - u             :: Values of parameters to be used in simulations.
 - u_names       :: SCM names for parameters `u`.
 - namelist      :: namelist to use for simulation.
 - namelist_args :: Additional arguments passed to the TurbulenceConvection namelist.
 - uuid          :: uuid of SCM run
 - les           :: path to LES stats file, or NamedTuple with keywords {forcing_model, month, experiment, cfsite_number} needed to specify path. 
 Outputs:
 - output_dir   :: directory containing output data from the SCM run.
 - model_error   :: Boolean specifying whether the simulation failed.
"""
function run_SCM_handler(
    case_name::String,
    out_dir::String;
    u::Vector{FT},
    u_names::Vector{String},
    namelist::Union{Dict, Nothing} = nothing,
    namelist_args::Union{Vector, Nothing} = nothing,
    uuid::String = "01",
    les::Union{NamedTuple, String, Nothing} = nothing,
) where {FT <: AbstractFloat}
    model_error = false

    # fetch default namelist if not provided
    if isnothing(namelist)
        namelist = NameList.default_namelist(case_name)
    end

    namelist["meta"]["uuid"] = uuid
    # set output dir to `out_dir`
    namelist["output"]["output_root"] = out_dir

    u_names, u = create_parameter_vectors(u_names, u)
    # Set optional namelist args
    if !isnothing(namelist_args)
        for namelist_arg in namelist_args
            change_entry!(namelist, namelist_arg)
        end
    end

    # update learnable parameter values
    @assert length(u_names) == length(u)
    for (pName, pVal) in zip(u_names, u)
        if haskey(namelist["turbulence"]["EDMF_PrognosticTKE"], pName)
            namelist["turbulence"]["EDMF_PrognosticTKE"][pName] = pVal
        elseif haskey(namelist["microphysics"], pName)
            namelist["microphysics"][pName] = pVal
        elseif haskey(namelist["time_stepping"], pName)
            namelist["time_stepping"][pName] = pVal
        else
            throw(
                ArgumentError("Parameter $pName cannot be calibrated. Consider adding namelist dictionary if needed."),
            )
        end
    end

    if case_name == "LES_driven_SCM"
        if isnothing(les)
            error("les path or keywords required for LES_driven_SCM case!")
        elseif isa(les, NamedTuple)
            les = get_stats_path(
                get_cfsite_les_dir(
                    les.cfsite_number;
                    forcing_model = les.forcing_model,
                    month = les.month,
                    experiment = les.experiment,
                ),
            )
        end
        namelist["meta"]["lesfile"] = les
    end

    # run TurbulenceConvection.jl with modified parameters
    logger = Logging.ConsoleLogger(stderr, Logging.Warn)
    _, ret_code = Logging.with_logger(logger) do
        main1d(namelist; time_run = false)
    end
    if ret_code ≠ :success
        model_error = true
        message = ["TurbulenceConvection.jl simulation $out_dir failed with parameters: \n"]
        append!(message, ["$param_name = $param_value \n" for (param_name, param_value) in zip(u_names, u)])
        @warn join(message)
    end

    return data_directory(out_dir, case_name, uuid), model_error
end

"""
    create_parameter_vectors(u_names::Vector{String}, u::Vector{FT}) where {FT <: AbstractFloat}

Given vector of parameter names and corresponding values, combine any vector components
into single parameter vectors for input into SCM.

Inputs:

 - `u_names` :: SCM names for parameters `u`, which may contain vector components.
 - `u` :: Values of parameters to be used in simulations, which may contain vector components.

Outputs:

 -  `u_names_out` :: SCM names for parameters `u`.
 -  `u_out` :: Values of parameters to be used in simulations.
"""
function create_parameter_vectors(u_names::Vector{String}, u::Vector{FT}) where {FT <: AbstractFloat}

    u_names_out = String[]
    u_out = []

    vector_param_inds = occursin.(r"{?}", u_names)
    pv_name_elem = rsplit.(u_names[vector_param_inds], "_", limit = 2) # get param name and element index
    u_vec_names, uvi = ~isempty(pv_name_elem) ? eachrow(reduce(hcat, pv_name_elem)) : ([], [])  # "transpose" `pv_name_elem`
    u_vec_inds = @. parse(Int64, only(split(uvi, (('{', '}'),), keepempty = false)))  # get `i` from "{i}" as Int64

    # collect scalar parameters
    scalar_param_inds = .!vector_param_inds
    append!(u_names_out, u_names[scalar_param_inds])
    append!(u_out, u[scalar_param_inds])

    # collect vector parameters
    for u_name in unique(u_vec_names)
        u_name_inds = u_vec_names .== u_name
        u_vals = u[vector_param_inds][u_name_inds]
        u_vals_sort_inds = u_vec_inds[u_name_inds]
        permute!(u_vals, u_vals_sort_inds)
        push!(u_names_out, u_name)
        push!(u_out, u_vals)
    end

    return u_names_out, u_out
end

"""
    generate_scm_input(
        model_evaluators::Vector{ModelEvaluator{FT}},
        outdir_path::String = pwd(),
    ) where {FT <: AbstractFloat}

Writes to file a set of ModelEvaluator used to initialize SCM
evaluations at different parameter vectors, as well as their
assigned numerical version.
"""
function generate_scm_input(
    model_evaluators::Vector{ModelEvaluator{FT}},
    outdir_path::String = pwd(),
    batch_indices::Union{Vector{Int}, Nothing} = nothing,
) where {FT <: AbstractFloat}
    # Generate versions conditioned on being unique within the batch.
    used_versions = Vector{Int}()
    for model_evaluator in model_evaluators
        version = rand(11111:99999)
        while version in used_versions
            version = rand(11111:99999)
        end
        jldsave(scm_init_path(outdir_path, version); model_evaluator, version, batch_indices)
        push!(used_versions, version)
    end
    return used_versions
end

"""
    get_gcm_les_uuid(
        cfsite_number::Integer;
        forcing_model::String,
        month::Integer,
        experiment::String,)

Generate unique and self-describing uuid given information about a GCM-driven LES
simulation from [Shen2022](@cite).
"""
function get_gcm_les_uuid(
    cfsite_number::Integer;
    forcing_model::String = "HadGEM2-A",
    month::Integer = 7,
    experiment::String = "amip",
)
    cfsite_number = string(cfsite_number)
    month = string(month, pad = 2)
    return join([cfsite_number, forcing_model, month, experiment], '_')
end

""" Save full EDMF data from every ensemble"""
function save_full_ensemble_data(save_path, sim_dirs_arr, ref_models)
    # get a simulation directory `.../Output.SimName.UUID`, and corresponding parameter name
    for (ens_i, sim_dirs) in enumerate(sim_dirs_arr)  # each ensemble returns a list of simulation directories
        ens_i_path = joinpath(save_path, "ens_$ens_i")
        mkpath(ens_i_path)
        for (ref_model, sim_dir) in zip(ref_models, sim_dirs)
            scm_name = ref_model.case_name
            # Copy simulation data to output directory
            dirname = splitpath(sim_dir)[end]
            @assert dirname[1:7] == "Output."  # sanity check
            # Stats file
            tmp_data_path = joinpath(sim_dir, "stats/Stats.$scm_name.nc")
            save_data_path = joinpath(ens_i_path, "Stats.$scm_name.$ens_i.nc")
            cp(tmp_data_path, save_data_path)
            # namefile
            tmp_namefile_path = namelist_directory(sim_dir, scm_name)
            save_namefile_path = namelist_directory(ens_i_path, scm_name)
            cp(tmp_namefile_path, save_namefile_path)
        end
    end
end

"""
    precondition(
        param::Vector{FT},
        priors,
        ref_models::Vector{ReferenceModel},
        ref_stats::ReferenceStatistics,
        namelist_args = nothing;
        counter::Integer = 0,
        max_counter::Integer = 10,
    ) where {FT <: Real}

Substitute parameter vector `param` by a parameter vector drawn
from the same prior, conditioned on the forward model being stable.

Inputs:

 - `param`      :: A parameter vector that may possibly result in unstable
    forward model evaluations (in unconstrained space).
 - `priors`      :: Priors from which the parameters were drawn.
 - `ref_models`  :: Vector of ReferenceModels to check stability for.
 - `ref_stats`   :: ReferenceStatistics of the ReferenceModels.
 - `namelist_args` :: Arguments passed to the TC.jl namelist.
 - `counter` :: Accumulator tracking number of recursive calls to preconditioner.
 - `max_counter` :: Maximum number of recursive calls to the preconditioner.

Outputs:

 - `new_param` :: A new parameter vector drawn from the prior, conditioned on simulations being stable (in unconstrained space).
"""
function precondition(
    param::Vector{FT},
    priors::ParameterDistribution,
    ref_models::Vector{ReferenceModel},
    ref_stats::ReferenceStatistics,
    namelist_args = nothing;
    counter::Integer = 0,
    max_counter::Integer = 10,
) where {FT <: Real}
    param_names = priors.name
    # Wrapper around SCM
    g_(u::Array{Float64, 1}) =
        run_SCM(u, param_names, ref_models, ref_stats, error_check = true, namelist_args = namelist_args)

    param_cons = deepcopy(transform_unconstrained_to_constrained(priors, param))
    _, _, _, model_error = g_(param_cons)
    if model_error && counter < max_counter
        message = ["Unstable parameter vector found: \n"]
        append!(message, ["$param_name = $param \n" for (param_name, param) in zip(param_names, param_cons)])
        @warn join(message)
        @warn "Sampling new parameter vector from prior..."
        return precondition(
            vec(construct_initial_ensemble(priors, 1)),
            priors,
            ref_models,
            ref_stats,
            namelist_args,
            counter = counter + 1,
            max_counter = max_counter,
        )
    elseif model_error
        @error "Number of recursive calls to preconditioner exceeded $(max_counter). Returning last failed parameter."
        return param
    else
        @info "Preconditioning finished."
        return param
    end
end

"""
    precondition(ME::ModelEvaluator, priors)

Substitute the parameter vector of a ModelEvaluator by another
one drawn from the given `priors`, conditioned on the forward
model being stable.

Inputs:
 - ME          :: A ModelEvaluator.
 - priors      :: Priors from which the parameters were drawn.
Outputs:
 - A preconditioned ModelEvaluator.
"""
function precondition(ME::ModelEvaluator, priors; namelist_args = nothing)
    # Precondition in unconstrained space
    u_orig = transform_constrained_to_unconstrained(priors, ME.param_cons)
    u = precondition(u_orig, priors, ME.ref_models, ME.ref_stats, namelist_args)
    # Transform back to constrained space
    param_cons = transform_unconstrained_to_constrained(priors, u)
    return ModelEvaluator(param_cons, ME.param_names, ME.ref_models, ME.ref_stats)
end

end # module
