module ReferenceModels

using JLD2
using ..HelperFuncs
using Random
using DocStringExtensions

import JSON
import TurbulenceConvection
tc = pkgdir(TurbulenceConvection)
include(joinpath(tc, "driver", "main.jl"))
include(joinpath(tc, "driver", "generate_namelist.jl"))
export main1d

import NCDatasets
NC = NCDatasets

export ReferenceModel, ReferenceModelBatch
export get_t_start, get_t_end, get_t_start_Σ, get_t_end_Σ, get_z_obs
export y_dir, Σ_dir, scm_dir, num_vars, uuid
export y_nc_file, Σ_nc_file, get_scm_namelist
export data_directory, namelist_directory
export construct_reference_models
export get_minibatch!, reshuffle_on_epoch_end, write_ref_model_batch
export time_shift_reference_model, write_val_ref_model_batch

"""
    ReferenceModel{FT <: Real}

A structure containing information about the 'true' reference model
and the observation map used to compare the parameterized and
reference models.

# Fields

$(TYPEDFIELDS)

# Constructors

    ReferenceModel(
        y_names::Vector{String},
        y_dir::String,
        scm_dir::String,
        case_name::String,
        t_start::Real,
        t_end::Real;
        Σ_dir::Union{String, Nothing} = nothing,
        Σ_t_start::Union{Real, Nothing} = nothing,
        Σ_t_end::Union{Real, Nothing} = nothing,
        n_obs::Union{Integer, Nothing} = nothing,
        namelist_args = nothing,
    )

`ReferenceModel` constructor allowing for any or all of `Σ_dir`, `Σ_t_start`, `Σ_t_end` to be
unspecified, in which case they take their values from `y_dir`, `t_start` and `t_end`, respectively.

    ReferenceModel(
        y_names::Vector{String},
        y_dir::String,
        scm_parent_dir::String,
        scm_suffix::String,
        case_name::String,
        t_start::Real,
        t_end::Real;
        Σ_dir::Union{String, Nothing} = nothing,
        Σ_t_start::Union{Real, Nothing} = nothing,
        Σ_t_end::Union{Real, Nothing} = nothing,
        n_obs::Union{Integer, Nothing} = nothing,
        namelist_args = nothing,
    )

`ReferenceModel` constructor using `scm_parent_dir`, `case_name`, `scm_suffix`, to define `scm_dir`.
"""
Base.@kwdef struct ReferenceModel{FT <: Real}
    "Vector of reference variable names"
    y_names::Vector{String}
    "Directory for reference data to compute `y` mean vector"
    y_dir::String
    "Directory for reference data to compute `Σ` covariance matrix"
    Σ_dir::String
    "Directory for static data related to forward scm model (parameter file & vertical levels)"
    scm_dir::String
    "Name of case"
    case_name::String
    # TODO: Make t_start and t_end vectors for multiple time intervals per reference model.
    "Start time for computing mean statistics over"
    y_t_start::FT
    "End time for computing mean statistics over"
    y_t_end::FT
    "Start time for computing covariance statistics over"
    Σ_t_start::FT
    "End time for computing covariance statistics over"
    Σ_t_end::FT
    "Vector of observed vertical locations"
    z_obs::Vector{FT}
    "TurbulenceConvection namelist"
    namelist::Dict
end  # ReferenceModel struct

function ReferenceModel(
    y_names::Vector{String},
    y_dir::String,
    scm_dir::String,
    case_name::String,
    t_start::Real,
    t_end::Real;
    Σ_dir::Union{String, Nothing} = nothing,
    Σ_t_start::Union{Real, Nothing} = nothing,
    Σ_t_end::Union{Real, Nothing} = nothing,
    n_obs::Union{Integer, Nothing} = nothing,
    namelist_args = nothing,
)
    # Always create new namelist
    namelist = get_scm_namelist(scm_dir, case_name, y_dir = y_dir, namelist_args = namelist_args, overwrite = true)
    z_obs = construct_z_obs(namelist)
    z_obs = !isnothing(n_obs) ? Array(LinRange(z_obs[1], z_obs[end], n_obs)) : z_obs
    FT = eltype(z_obs)

    Σ_dir = isnothing(Σ_dir) ? y_dir : Σ_dir
    Σ_t_start = isnothing(Σ_t_start) ? t_start : Σ_t_start
    Σ_t_end = isnothing(Σ_t_end) ? t_end : Σ_t_end

    ReferenceModel{FT}(
        y_names,
        y_dir,
        Σ_dir,
        scm_dir,
        case_name,
        FT(t_start),
        FT(t_end),
        FT(Σ_t_start),
        FT(Σ_t_end),
        z_obs,
        namelist,
    )
end

function ReferenceModel(
    y_names::Vector{String},
    y_dir::String,
    scm_parent_dir::String,
    scm_suffix::String,
    case_name::String,
    t_start::Real,
    t_end::Real;
    Σ_dir::Union{String, Nothing} = nothing,
    Σ_t_start::Union{Real, Nothing} = nothing,
    Σ_t_end::Union{Real, Nothing} = nothing,
    n_obs::Union{Integer, Nothing} = nothing,
    namelist_args = nothing,
)
    scm_dir = data_directory(scm_parent_dir, case_name, scm_suffix)
    args = (y_names, y_dir, scm_dir, case_name, t_start, t_end)
    return ReferenceModel(
        args...,
        Σ_dir = Σ_dir,
        Σ_t_start = Σ_t_start,
        Σ_t_end = Σ_t_end,
        n_obs = n_obs,
        namelist_args = namelist_args,
    )
end

get_t_start(m::ReferenceModel) = m.y_t_start
get_t_end(m::ReferenceModel) = m.y_t_end
get_t_start_Σ(m::ReferenceModel) = m.Σ_t_start
get_t_end_Σ(m::ReferenceModel) = m.Σ_t_end

"Returns the observed vertical locations for a reference model"
get_z_obs(m::ReferenceModel) = m.z_obs

y_dir(m::ReferenceModel) = m.y_dir
Σ_dir(m::ReferenceModel) = m.Σ_dir
scm_dir(m::ReferenceModel) = m.scm_dir
get_scm_namelist(m::ReferenceModel) = deepcopy(m.namelist)

# TODO: cache filename and move `get_stats_path` call to constructor.
y_nc_file(m::ReferenceModel) = get_stats_path(y_dir(m))
Σ_nc_file(m::ReferenceModel) = get_stats_path(Σ_dir(m))

data_directory(root::S, name::S, suffix::S) where {S <: AbstractString} = joinpath(root, "Output.$name.$suffix")
uuid(m::ReferenceModel) = String(split(scm_dir(m), ".")[end])

namelist_directory(root::String, m::ReferenceModel) = namelist_directory(root, m.case_name)
namelist_directory(root::S, casename::S) where {S <: AbstractString} = joinpath(root, "namelist_$casename.in")

num_vars(m::ReferenceModel) = length(m.y_names)

"""
    get_scm_namelist(
        output_dir::String,
        case_name::String;
        y_dir::Union{String, Nothing} = nothing,
        overwrite::Bool = false,
        namelist_args = nothing,
    )::Dict

Returns a TurbulenceConvection.jl namelist, given the case and a list of namelist arguments.

Inputs:

 - `output_dir`     :: Directory where the namelist will be stored or should be located.
 - `case_name`      :: Name of the TurbulenceConvection.jl case considered.
 - `y_dir`          :: Directory with LES data to drive the SCM with, if `case_name` is `LES_driven_SCM`.
 - `overwrite`      :: Whether to overwrite the namelist file, if it exists.
 - `namelist_args`  :: Vector of non-default arguments to be used in the namelist, defined as a vector of tuples.
Outputs:
 - `namelist`   :: The TurbulenceConvection.jl namelist.
"""
function get_scm_namelist(
    output_dir::String,
    case_name::String;
    y_dir::Union{String, Nothing} = nothing,
    overwrite::Bool = true,
    namelist_args = nothing,
)::Dict
    namelist_path = namelist_directory(output_dir, case_name)
    namelist = if ~isfile(namelist_path) | overwrite
        NameList.default_namelist(case_name, root = output_dir)
    else
        JSON.parsefile(namelist_path)
    end
    namelist["stats_io"]["calibrate_io"] = true
    if !isnothing(namelist_args)
        for namelist_arg in namelist_args
            change_entry!(namelist, namelist_arg)
        end
    end

    namelist["meta"]["uuid"] = String(split(output_dir, ".")[end])
    namelist["output"]["output_root"] = dirname(output_dir)
    # if `LES_driven_SCM` case, provide input LES stats file
    if case_name == "LES_driven_SCM"
        @assert !isnothing(y_dir) "lesfile must be specified in the construction of LES_driven_SCM namelist."
        namelist["meta"]["lesfile"] = get_stats_path(y_dir)
    end

    return namelist
end

"""
    construct_z_obs(namelist::Dict)

Constructs the vector of observed locations given a TurbulenceConvection.jl namelist.
"""
function construct_z_obs(namelist::Dict)
    grid = construct_grid(namelist)
    return vec(grid.zc)
end

"""
    construct_reference_models(kwarg_ld::Dict{Symbol, Vector{T} where T})::Vector{ReferenceModel}

Returns a vector of `ReferenceModel`s given a dictionary of keyword argument lists.

Inputs:

 - `kwarg_ld`     :: Dictionary of keyword argument lists

Outputs:

 - `ref_models`   :: Vector where the i-th ReferenceModel is constructed from the i-th element of every keyword argument list of the dictionary.
"""
function construct_reference_models(kwarg_ld::Dict{Symbol, Vector{T} where T})::Vector{ReferenceModel}
    n_RM = length(kwarg_ld[:case_name])
    ref_models = Vector{ReferenceModel}()
    for RM_i in 1:n_RM
        kw = Dict(k => v[RM_i] for (k, v) in pairs(kwarg_ld))  # unpack dict
        args = if haskey(kw, :scm_dir)
            # construct ref_models using `scm_dir` directly
            (kw[j] for j in (:y_names, :y_dir, :scm_dir, :case_name, :t_start, :t_end))
        elseif all(haskey.(Ref(kw), [:scm_parent_dir, :scm_suffix]))
            # construct ref_models using `scm_parent_dir` and `scm_suffix`
            (kw[j] for j in (:y_names, :y_dir, :scm_parent_dir, :scm_suffix, :case_name, :t_start, :t_end))
        else
            throw(
                ArgumentError(
                    "You need to specify either `scm_dir` or all of " *
                    "(`scm_parent_dir`, `case_name`, `scm_suffix`) to construct a `ReferenceModel`",
                ),
            )
        end
        push!(
            ref_models,
            ReferenceModel(
                args...,
                Σ_dir = get(kw, :Σ_dir, nothing),
                Σ_t_start = get(kw, :Σ_t_start, nothing),
                Σ_t_end = get(kw, :Σ_t_end, nothing),
                n_obs = get(kw, :n_obs, nothing),
                namelist_args = get(kw, :namelist_args, nothing),
            ),
        )
    end
    return ref_models
end

"""
    time_shift_reference_model(m::ReferenceModel, Δt::FT) where {FT <: Real}

Returns a time-shifted ReferenceModel, considering an interval relative to the last
available time step of the original model.

Inputs:

 - `m`     :: A ReferenceModel.
 - `Δt`  :: [LES last time - SCM start time (LES timeframe)]

Outputs:

 - The time-shifted ReferenceModel.
"""
function time_shift_reference_model(m::ReferenceModel, Δt::FT) where {FT <: Real}
    filename = y_nc_file(m)
    t = nc_fetch(filename, "t")
    t_start = t[end] - Δt + get_t_start(m)
    t_end = t[end] - Δt + get_t_end(m)
    Σ_t_start = t[end] - Δt + get_t_start_Σ(m)
    Σ_t_end = t[end] - Δt + get_t_end_Σ(m)

    @assert t_start >= 0 "t_start must be positive after time shift, but $t_start was given."
    @assert Σ_t_start >= 0 "Σ_t_start must be positive after time shift, but $Σ_t_start was given."
    @info string(
        "Shifting time windows for ReferenceModel $(m.case_name)",
        "to ty=($t_start, $t_end), tΣ=($Σ_t_start, $Σ_t_end).",
    )
    return ReferenceModel(
        m.y_names,
        m.y_dir,
        m.Σ_dir,
        m.scm_dir,
        m.case_name,
        t_start,
        t_end,
        Σ_t_start,
        Σ_t_end,
        m.z_obs,
        m.namelist,
    )
end

"""
    struct ReferenceModelBatch

A structure containing a batch of ReferenceModels and an evaluation
order for ReferenceModels within the current epoch.

# Fields

$(TYPEDFIELDS)

# Constructors
    
    ReferenceModelBatch(ref_models::Vector{ReferenceModel}, shuffling::Bool = true)

`ReferenceModelBatch` constructor given a vector of `ReferenceModel`s.


    ReferenceModelBatch(kwarg_ld::Dict{Symbol, Vector{T} where T}, shuffling::Bool = true)

`ReferenceModelBatch` constructor given a dictionary of keyword argument lists.

Inputs:

 - `kwarg_ld`     :: Dictionary of keyword argument lists
 - `shuffling`    :: Whether to shuffle the order of ReferenceModels.
"""
Base.@kwdef struct ReferenceModelBatch
    "Vector containing all reference models"
    ref_models::Vector{ReferenceModel}
    "Vector of indices defining the `ReferenceModel` evaluation order when batching"
    eval_order::Vector{Int}
end

function ReferenceModelBatch(kwarg_ld::Dict{Symbol, Vector{T} where T}, shuffling::Bool = true)
    ref_models = construct_reference_models(kwarg_ld)
    eval_order = shuffling ? shuffle(1:length(ref_models)) : 1:length(ref_models)
    return ReferenceModelBatch(ref_models, eval_order)
end

function ReferenceModelBatch(ref_models::Vector{ReferenceModel}, shuffling::Bool = true)
    eval_order = shuffling ? shuffle(1:length(ref_models)) : 1:length(ref_models)
    return ReferenceModelBatch(ref_models, eval_order)
end

"""
    get_minibatch!(ref_models::ReferenceModelBatch, batch_size::Int)

Returns a minibatch of `ReferenceModel`s from a ReferenceModelBatch and updates
the eval order.

The size of the minibatch is either the requested size, or the remainder of the
elements in the eval_order for this epoch.

Inputs:

 - `ref_model_batch` :: A ReferenceModelBatch.
 - `batch_size`      :: The number of `ReferenceModel`s to retrieve.

Outputs:

 - A vector of `ReferenceModel`s.
 - The indices of the returned `ReferenceModel`s.
"""
function get_minibatch!(ref_model_batch::ReferenceModelBatch, batch_size::Int)
    batch = min(batch_size, length(ref_model_batch.eval_order))
    indices = [pop!(ref_model_batch.eval_order) for i in 1:batch]
    return ref_model_batch.ref_models[indices], indices
end

"Restarts a shuffled evaluation order if the current epoch has finished."
function reshuffle_on_epoch_end(ref_model_batch::ReferenceModelBatch, shuffling::Bool = true)
    if isempty(ref_model_batch.eval_order)
        @info "Current epoch finished. Reshuffling dataset."
        return ReferenceModelBatch(ref_model_batch.ref_models, shuffling)
    else
        return ref_model_batch
    end
end

function write_ref_model_batch(ref_model_batch::ReferenceModelBatch; outdir_path::String = pwd())
    jldsave(joinpath(outdir_path, "ref_model_batch.jld2"); ref_model_batch)
end
function write_val_ref_model_batch(ref_model_batch::ReferenceModelBatch; outdir_path::String = pwd())
    jldsave(joinpath(outdir_path, "val_ref_model_batch.jld2"); ref_model_batch)
end

end # module
