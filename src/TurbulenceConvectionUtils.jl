module TurbulenceConvectionUtils

export ModelEvaluator,
    run_SCM,
    run_SCM_handler,
    run_reference_SCM,
    generate_scm_input,
    parse_version_inds,
    eval_single_ref_model,
    save_full_ensemble_data,
    precondition,
    get_gcm_les_uuid,
    save_tc_data

import Logging
using JLD2
using JSON
using Random
using DocStringExtensions
using TurbulenceConvection
tc = pkgdir(TurbulenceConvection)
include(joinpath(tc, "driver", "main.jl"))

# EKP modules
using EnsembleKalmanProcesses.ParameterDistributions
import EnsembleKalmanProcesses: construct_initial_ensemble

using ..ReferenceModels
import ..ReferenceModels: NameList

using ..ReferenceStats
using ..HelperFuncs
using ..AbstractTypes
import ..AbstractTypes: OptVec

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
    "A mapping operator to define relations between parameters.
     See [`ParameterMap`](@ref) for details."
    param_map::ParameterMap
    "Vector of reference models"
    ref_models::Vector{ReferenceModel}
    "Reference statistics for the inverse problem"
    ref_stats::ReferenceStatistics

    function ModelEvaluator(
        param_cons::Vector{FT},
        param_names::Vector{String},
        param_map::ParameterMap,
        RM::Vector{ReferenceModel},
        RS::ReferenceStatistics,
    ) where {FT <: Real}
        return new{FT}(param_cons, param_names, param_map, RM, RS)
    end
end


"""
    run_SCM(u, u_names, param_map, RM::Vector{ReferenceModel}, RS::ReferenceStatistics; 
            [error_check = false, failure_handler = "high_loss"])
    run_SCM(ME::ModelEvaluator; [error_check = false, failure_handler = "high_loss"])

Run the single-column model (SCM) using a set of parameters and return the value of outputs defined in y_names, possibly
after normalization and projection onto lower dimensional space using PCA.

The function also outputs a boolean diagnosing whether the requested SCM simulation failed.

# Arguments
- `u`           :: Values of parameters to be used in simulations.
- `u_names`     :: SCM names for parameters `u`.
- `param_map`   :: A mapping operator to define relations between parameters. See also [`ParameterMap`](@ref).
- `RM`          :: Vector of `ReferenceModel`s
- `RS`          :: Reference statistics for simulation

# Keywords
- `error_check`     :: Returns as an additional argument whether the SCM call errored.
- `failure_handler` :: Method used to handle failed simulations.

# Returns
- `sim_dirs`    :: Vector of simulation output directories
- `g_scm`       :: Vector of model evaluations concatenated for all flow configurations.
- `g_scm_pca`   :: Projection of `g_scm` onto principal subspace spanned by eigenvectors.
- `model_error` :: Whether the simulation errored with the requested configuration.
"""
function run_SCM(
    u::Vector{FT},
    u_names::Vector{<:AbstractString},
    param_map::ParameterMap,
    RM::Vector{ReferenceModel},
    RS::ReferenceStatistics{FT};
    error_check::Bool = false,
    failure_handler = "high_loss",
) where {FT <: Real}

    mkpath(joinpath(pwd(), "tmp"))
    result_arr = map(x -> eval_single_ref_model(x..., RS, u, u_names, param_map), enumerate(RM))
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
function run_SCM(ME::ModelEvaluator; error_check::Bool = false, failure_handler = "high_loss")
    return run_SCM(
        ME.param_cons,
        ME.param_names,
        ME.param_map,
        ME.ref_models,
        ME.ref_stats,
        error_check = error_check,
        failure_handler = failure_handler,
    )
end

"""
    eval_single_ref_model(m_index, m::ReferenceModel, RS::ReferenceStatistics, u, u_names, param_map)

Run the single-column model (SCM) under a single configuration (i.e., [`ReferenceModel`](@ref)) 
using a list of parameters `u` with names `u_names`, and return the forward model evaluation in both the original 
and the latent PCA space.

# Arguments
- `m_index`       :: Index of `m` in within the list of overarching `ReferenceModel`s vector used to construct `RS`.
- `m`             :: A [`ReferenceModel`](@ref).
- `RS`            :: The [`ReferenceStatistics`](@ref) for simulation.
- `u`             :: Values of parameters to be used in simulations.
- `u_names`       :: SCM names corresponding to parameter values `u`.
- `param_map`     :: A mapping operator to define relations between parameters. See also [`ParameterMap`](@ref).

# Returns
- `sim_dir::String`     :: Simulation output directory.
- `g_scm::Vector`       :: Forward model evaluation in original output space.
- `g_scm_pca::Vector`   :: Projection of `g_scm` onto principal subspace spanned by eigenvectors.
- `model_error::Bool`   :: Whether the simulation errored with the requested configuration.
"""
function eval_single_ref_model(
    m_index::Integer,
    m::ReferenceModel,
    RS::ReferenceStatistics,
    u::Vector{FT},
    u_names::Vector{<:AbstractString},
    param_map::ParameterMap,
) where {FT <: Real}
    # create temporary directory to store SCM data in
    tmpdir = mktempdir(joinpath(pwd(), "tmp"))
    # run TurbulenceConvection.jl. Get output directory for simulation data
    sim_dir, model_error = run_SCM_handler(m, tmpdir, u, u_names, param_map)
    filename = get_stats_path(sim_dir)
    z_obs = get_z_obs(m)
    if model_error
        d_full, d = size(RS.pca_vec[m_index])
        g_scm = fill(NaN, d_full)
        g_scm_pca = fill(NaN, d)
    else
        g_scm, prof_indices = get_profile(m, filename, z_scm = z_obs, prof_ind = true)
        g_scm = normalize_profile(g_scm, RS.norm_vec[m_index], length(z_obs), prof_indices)
        # perform PCA reduction
        g_scm_pca = RS.pca_vec[m_index]' * g_scm
    end
    return sim_dir, g_scm, g_scm_pca, model_error
end

"""
    run_reference_SCM(m::ReferenceModel; [overwrite = false, run_single_timestep = true])

Run the single-column model (SCM) for a reference model object using default parameters, and write the output to file.

# Arguments:
- `m`                    :: A [`ReferenceModel`](@ref).
 
# Keywords
- `output_root`         :: Directory within which the simulation results folder is saved to.
- `uuid`                :: Simulation identifier. Appended to the simulation results folder.
- `overwrite`           :: If `true`, run TC.jl and overwrite existing simulation files.
- `run_single_timestep` :: If `true`, run only one time step.
"""
function run_reference_SCM(
    m::ReferenceModel;
    output_root::AbstractString = pwd(),
    uuid::AbstractString = "01",
    overwrite::Bool = false,
    run_single_timestep::Bool = true,
)
    output_dir = data_directory(output_root, m.case_name, uuid)
    if ~isdir(joinpath(output_dir, "stats")) | overwrite
        namelist = get_scm_namelist(m)

        namelist["output"]["output_root"] = output_root
        namelist["meta"]["uuid"] = uuid

        if run_single_timestep
            # Run only 1 timestep -- since we don't need output data, only simulation config
            namelist["time_stepping"]["adapt_dt"] = false
            namelist["time_stepping"]["t_max"] = namelist["time_stepping"]["dt_min"]
        end
        # run TurbulenceConvection.jl
        logger = Logging.ConsoleLogger(stderr, Logging.Warn)
        _, _, ret_code = Logging.with_logger(logger) do
            main1d(namelist; time_run = false)
        end
        if ret_code ≠ :success
            @warn "Default TurbulenceConvection.jl simulation $(basename(m.y_dir)) failed."
        end
    end
end

"""
    run_SCM_handler(case_name::String, out_dir, u, u_names, param_map; [namelist, namelist_args, uuid, les])
    run_SCM_handler(m::ReferenceModel, out_dir, u, u_names, param_map)

Run a TurbulenceConvection.jl case. Return directory path where simulation data is stored pointing and a flag for whether the simulation failed.

The case is specified with the first function argument, either by providing the case name as a `String`,
or by passing a [`ReferenceModel`](@ref) `m`. The `out_dir` is the path where the output directory is 
to be stored. The returned path is therefore of the form `out_dir/Output.case_name.uuid` where `uuid`
is a simulation run identifier (defaults to "01"). 

The keywords `u` and `u_names` are vectors of parameter values and names that will be modified in the TC.jl run. 

If a `case_name` is provided, a `namelist` can optionally be provided that defines all case-specific 
parameters and run-options needed to run TC.jl. If this not provided, the default case-specific 
namelist from TC.jl is used. Additional arguments to modify the TC.jl namelist can 
also be provided to `namelist_args`; see also [`change_entry!`](@ref).

The [`ParameterMap`](@ref) `param_map` sets the values of specific parameters provided pre-defined mapping operators. 
See also [`ParameterMap`](@ref).

If the case is `LES_driven_SCM`, a path to an LES stats file must be provided to `les`. 
See also [`get_cfsite_les_dir`](@ref).

Note that if a [`ReferenceModel`](@ref) `m` is passed, `namelist`, `uuid` and `les` are not valid keywords.
In this case, `namelist` and `les` are fetched from the `ReferenceModel`, and `uuid` is a random string.

# Arguments
- `m`               :: A [`ReferenceModel`](@ref)
- `case_name`       :: Name of case to run. For example: `Bomex`, `TRMM_LBA`, `LES_driven_SCM`.
- `out_dir`         :: Path to where folder with simulation results are to be saved.
- `u`               :: Values of parameters to be used in simulations.
- `u_names`         :: SCM names for parameters `u`.
- `param_map`       :: A mapping operator to define relations between parameters. See [`ParameterMap`](@ref) for details.

# Keywords
- `namelist`        :: namelist to use for simulation.
- `uuid`            :: uuid of SCM run
- `les`             :: path to LES stats file, or NamedTuple with keywords {forcing_model, month, experiment, cfsite_number} needed to specify path. 

# Returns
- `String`          :: Directory path where output data from the TC.jl run is stored.
- `Bool`            :: `true` if the simulation failed/crashed, `false` otherwise.
"""
function run_SCM_handler(
    case_name::AbstractString,
    out_dir::AbstractString,
    u::Vector{<:Real},
    u_names::Vector{<:AbstractString},
    param_map::ParameterMap;
    namelist::Dict,
    uuid::AbstractString = "01",
    les::Union{NamedTuple, AbstractString, Nothing} = nothing,
)
    model_error = false

    namelist["meta"]["uuid"] = uuid
    # set output dir to `out_dir`
    namelist["output"]["output_root"] = out_dir

    u_names, u = create_parameter_vectors(u_names, u, param_map, namelist)

    # update learnable parameter values
    @assert length(u_names) == length(u)
    for (pName, pVal) in zip(u_names, u)
        param_subdict = namelist_subdict_by_key(namelist, pName)
        param_subdict[pName] = pVal
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
    _, _, ret_code = Logging.with_logger(logger) do
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

function run_SCM_handler(
    m::ReferenceModel,
    out_dir::AbstractString,
    u::Vector{<:Real},
    u_names::Vector{<:AbstractString},
    param_map::ParameterMap,
)

    # fetch namelist
    namelist = get_scm_namelist(m)
    # output subset of variables needed for calibration
    namelist["stats_io"]["calibrate_io"] = true
    # set random uuid
    uuid = randstring(RandomDevice(), 5)

    # run TurbulenceConvection.jl with modified parameters
    return run_SCM_handler(
        m.case_name,
        out_dir,
        u,
        u_names,
        param_map;
        namelist = namelist,
        namelist_args = namelist_args,
        uuid = uuid,
        les = get(namelist["meta"], "lesfile", nothing),
    )
end

"""
    create_parameter_vectors(u_names, u, param_map, namelist)

Given vector of parameter names and corresponding values, combine any vector components into single parameter vectors for input into SCM.

Arguments:
- `u_names`    :: SCM names for parameters `u`, which may contain vector components.
- `u`          :: Values of parameters to be used in simulations, which may contain vector components.
- `param_map`  :: A mapping to a reduced parameter set. See [`ParameterMap`](@ref) for details.
- `namelist`   :: The parameter namelist for TurbulenceConvection.jl

Returns:
-  `Vector{String}`         :: SCM names for parameters `u`.
-  `Vector{Vector{Real}}`   :: Values of parameters to be used in simulations.
"""
function create_parameter_vectors(
    u_names::Vector{<:AbstractString},
    u::Vector{<:Real},
    param_map::ParameterMap,
    namelist::Dict,
)
    # Apply the `param_map` from the calibrated parameters to all parameters.
    u_names, u = expand_params(u_names, u, param_map, namelist)

    u_names_out = String[]
    u_out = []

    find_vec_params = match.(r"(.+)_{(\d+)}", u_names)
    filter_vec_params = filter(!isnothing, find_vec_params)
    u_vec_names = getindex.(filter_vec_params, 1)
    u_vec_inds = getindex.(filter_vec_params, 2)

    # collect scalar parameters
    scalar_param_inds = isnothing.(find_vec_params)
    append!(u_names_out, u_names[scalar_param_inds])
    append!(u_out, u[scalar_param_inds])

    # collect vector parameters
    vector_param_inds = @. !isnothing(find_vec_params)
    for u_name in unique(u_vec_names)
        u_name_inds = u_vec_names .== u_name
        u_vals = u[vector_param_inds][u_name_inds]
        u_vals_sort_inds = sortperm(u_vec_inds[u_name_inds])
        permute!(u_vals, u_vals_sort_inds)
        push!(u_names_out, u_name)
        push!(u_out, u_vals)
    end
    return u_names_out, u_out
end

"""
    generate_scm_input(model_evaluators, iteration, [outdir_path = pwd(), batch_indices = nothing])

Writes to file a set of [`ModelEvaluator`](@ref) used to initialize SCM evaluations at different parameter vectors, 
as well as their assigned version, which is on the form `i<iteration>_e<ensemble_index>`.
"""
function generate_scm_input(
    model_evaluators::Vector{ModelEvaluator{FT}},
    iteration::Int,
    outdir_path::String = pwd(),
    batch_indices::OptVec{Int} = nothing,
) where {FT <: AbstractFloat}
    versions = String[]
    for (ens_i, model_evaluator) in enumerate(model_evaluators)
        version = "i$(iteration)_e$ens_i"
        jldsave(scm_init_path(outdir_path, version); model_evaluator, version, batch_indices)
        push!(versions, version)
    end
    return versions
end

""" 
    parse_version_inds(version)

Given `version = "ix_ey"`, return the iteration index `x` and ensemble index `y` as integers.
"""
parse_version_inds(version::String) = parse.(Int, SubString.(split(version, "_"), 2, lastindex.(split(version, "_"))))

"""
    get_gcm_les_uuid(cfsite_number; [forcing_model::String, month, experiment])

Generate unique and self-describing uuid given information about a GCM-driven LES simulation from [Shen2022](@cite).

# Examples
```
julia> get_gcm_les_uuid(1; forcing_model = "HadGEM2-A", month = 7, experiment = "amip")
"1_HadGEM2-A_07_amip"
```
"""
function get_gcm_les_uuid(cfsite_number; forcing_model, month::Integer, experiment)
    return "$(cfsite_number)_$(forcing_model)_$(string(month, pad = 2))_$(experiment)"
end

"""
    save_tc_data(cases, outdir_path, sim_dirs, version, suffix)

Save full TC.jl output in `<results_folder>/timeseries.<suffix>/iter_<iteration>/Output.<case>.<case_id>_<ens_i>`.

If the flag `save_tc_output` is set to true in the output config, i.e. `config["output"]`, TC.jl output is saved, 
and if additionally a list of iterations is specified in `save_tc_iterations` (also in the output config),
only those EKP iterations are saved.

# Arguments
- `config`      :: The calibration config dictionary.
- `outdir_path` :: The results `results_folder` path
- `sim_dirs`    :: List of (temporary) directories where raw simulation output is initially saved to
- `version`     :: An identifier for the current iteration and ensemble index
- `suffix`      :: Case set identifier; is either "train" or "validation".
"""
function save_tc_data(config, outdir_path, sim_dirs, version, suffix)
    cases = config["reference"]["case_name"]
    @assert length(cases) == length(sim_dirs) "The number of cases $(length(cases)) should equal the number of TC output directories $(length(sim_dirs))."
    N_iter = config["process"]["N_iter"]
    save_tc_iterations = get(config["output"], "save_tc_iterations", 1:N_iter)
    iter_ind, ens_ind = parse_version_inds(version)

    # Only save specified iterations
    if !(iter_ind ∈ save_tc_iterations)
        return
    end

    for (i, (case, sim_dir)) in enumerate(zip(cases, sim_dirs))
        version_dst_dir = joinpath(outdir_path, "timeseries.$suffix/iter_$iter_ind")
        mkpath(version_dst_dir)
        # ensure that simulation directory is unique given possibly identical case names
        case_id = length(cases[1:i][(cases .== case)[1:i]])
        if isdir(sim_dir)
            mv(sim_dir, joinpath(version_dst_dir, "Output.$case.$(case_id)_$ens_ind"))
        else
            @warn("sim directory not found: $sim_dir")
        end
    end  # end cases, sim_dirs
end


"""
    precondition(
        param::Vector{FT},
        priors,
        param_map::ParameterMap,
        ref_models::Vector{ReferenceModel},
        ref_stats::ReferenceStatistics;
        counter::Integer = 0,
        max_counter::Integer = 10,
    ) where {FT <: Real}

Substitute parameter vector `param` by a parameter vector drawn
from the same prior, conditioned on the forward model being stable.

Inputs:

 - `param`          :: A parameter vector that may possibly result in unstable
                        forward model evaluations (in unconstrained space).
 - `priors`         :: Priors from which the parameters were drawn.
 - `param_map`      :: A mapping to a reduced parameter set. See [`ParameterMap`](@ref) for details.
 - `ref_models`     :: Vector of ReferenceModels to check stability for.
 - `ref_stats`      :: ReferenceStatistics of the ReferenceModels.
 - `counter`        :: Accumulator tracking number of recursive calls to preconditioner.
 - `max_counter`    :: Maximum number of recursive calls to the preconditioner.

Outputs:

 - `new_param` :: A new parameter vector drawn from the prior, conditioned on simulations being stable (in unconstrained space).
"""
function precondition(
    param::Vector{FT},
    priors::ParameterDistribution,
    param_map::ParameterMap,
    ref_models::Vector{ReferenceModel},
    ref_stats::ReferenceStatistics;
    counter::Integer = 0,
    max_counter::Integer = 10,
) where {FT <: Real}
    param_names = priors.name
    # Wrapper around SCM
    g_(u::Array{Float64, 1}) = run_SCM(u, param_names, param_map, ref_models, ref_stats, error_check = true)

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
            param_map,
            ref_models,
            ref_stats;
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
function precondition(ME::ModelEvaluator, priors)
    # Precondition in unconstrained space
    u_orig = transform_constrained_to_unconstrained(priors, ME.param_cons)
    u = precondition(u_orig, priors, ME.param_map, ME.ref_models, ME.ref_stats)
    # Transform back to constrained space
    param_cons = transform_unconstrained_to_constrained(priors, u)
    return ModelEvaluator(param_cons, ME.param_names, ME.param_map, ME.ref_models, ME.ref_stats)
end

end # module
