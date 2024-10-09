module ReferenceModels

export ReferenceModel,
    ReferenceModelBatch,
    get_t_start,
    get_t_end,
    get_t_start_Σ,
    get_t_end_Σ,
    get_z_obs,
    get_z_rectified_obs, # i think this is needed for using custom z
    get_y_dir,
    get_Σ_dir,
    num_vars,
    uuid,
    y_nc_file,
    Σ_nc_file,
    get_scm_namelist,
    data_directory,
    namelist_directory,
    get_ref_model_kwargs,
    construct_reference_models,
    get_minibatch!,
    reshuffle_on_epoch_end,
    write_ref_model_batch,
    time_shift_reference_model,
    write_val_ref_model_batch

using JLD2
using Random
using DocStringExtensions

import JSON
import TurbulenceConvection
TC = TurbulenceConvection
tc = pkgdir(TC)
include(joinpath(tc, "driver", "Cases.jl"))
include(joinpath(tc, "driver", "common_spaces.jl"))
include(joinpath(tc, "driver", "generate_namelist.jl"))

import NCDatasets
NC = NCDatasets

using ..AbstractTypes
using ..ModelTypes
using ..HelperFuncs

import ..AbstractTypes: OptVec, OptInt, OptString, OptReal

"""
    ReferenceModel{FT <: Real}

A structure containing information about the 'true' reference model
and the observation map used to compare the parameterized and
reference models.

# Fields

$(TYPEDFIELDS)

# Constructors

    ReferenceModel(y_names, y_dir, case_name, t_start, t_end; [Σ_dir, Σ_t_start, Σ_t_end, y_type, Σ_type, n_obs, namelist_args, seed])

A [`ReferenceModel`](@ref) can be defined for a case `case_name`, provided the location of the data, `y_dir`, the 
reference variable names `y_names`, and the averaging interval (`t_start`, `t_end`) is provided.

If data and/or averaging intervals for the empirical covariance matrix `Σ` is different than the mean observations `y`,
this is specified with `Σ_dir`, `Σ_t_start`, and `Σ_t_end`.

`ReferenceModel` constructor allowing for any or all of `Σ_dir`, `Σ_t_start`, `Σ_t_end` to be
unspecified, in which case they take their values from `y_dir`, `t_start` and `t_end`, respectively.

A tuple of `namelist_args` can be specified to overwrite default arguments for the case in TurbulenceConvection.jl.

Mainly for testing purposes, a `seed` can also be specified to avoid randomness during namelist generation.

"""
Base.@kwdef struct ReferenceModel{FT <: Real}
    "Vector of reference variable names"
    y_names::Vector{String}
    "Directory for reference data to compute `y` mean vector"
    y_dir::String
    "Directory for reference data to compute `Σ` covariance matrix"
    Σ_dir::String
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
    "Vector of desired vertical locations for the comparison -- you may not wish to compare all z locations so we allow to to specify"
    z_rectified_obs::Vector{FT}
    "TurbulenceConvection namelist"
    namelist::Dict
    "Type of model used to generate mean observations"
    y_type::ModelType
    "Type of model used to generate observational noise"
    Σ_type::ModelType
end  # ReferenceModel struct

function ReferenceModel(
    y_names::Vector{String},
    y_dir::String,
    case_name::String,
    t_start::Real,
    t_end::Real;
    Σ_dir::OptString = nothing,
    Σ_t_start::OptReal = nothing,
    Σ_t_end::OptReal = nothing,
    y_type::ModelType = LES(),
    Σ_type::ModelType = LES(),
    n_obs::OptInt = nothing,
    namelist_args::OptVec{<:Tuple} = nothing,
    seed::OptInt = nothing,
    z_rectifier::Union{Function, NTuple{2, FT2}, Vector{FT2}, Nothing} = nothing,
) where {FT2 <: Real}
    if case_name == "LES_driven_SCM" || occursin("socrates", lowercase(case_name))
        @assert isa(y_type, LES) || isa(Σ_type, LES) "LES data must be used in the construction of LES_driven_SCM ReferenceModels."
    end
    les_dir = isa(y_type, LES) ? y_dir : Σ_dir

    # Always create new namelist
    namelist = get_scm_namelist(case_name, les_dir = les_dir, namelist_args = namelist_args, seed = seed)
    z_obs = construct_z_obs(namelist)
    z_obs = !isnothing(n_obs) ? Array(range(z_obs[1], z_obs[end], n_obs)) : z_obs
    FT = eltype(z_obs)

    if !isnothing(z_rectifier)
        if isa(z_rectifier, Vector{FT2}) # single z vector, use as is
            z_rectified_obs = z_rectifier
        elseif isa(z_rectifier, Function)
            z_rectified_obs = z_rectifier(z_obs) # apply some func
        elseif isa(z_rectifier, NTuple{2, FT2}) # select only z between z_rectifier[1] and z_rectifier[2]
            z_rectified_obs = z_obs[(z_obs .>= z_rectifier[1]) .& (z_obs .<= z_rectifier[2])]
        else
            throw(ArgumentError("z_rectifiers must be a function, a tuple of two floats, or a vector of floats."))
        end
    else
        z_rectified_obs = z_obs
    end
    z_rectified_obs = FT.(z_rectified_obs) # in case FT and FT2 were different

    Σ_dir = isnothing(Σ_dir) ? y_dir : Σ_dir
    Σ_t_start = isnothing(Σ_t_start) ? t_start : Σ_t_start
    Σ_t_end = isnothing(Σ_t_end) ? t_end : Σ_t_end

    ReferenceModel{FT}(
        y_names,
        y_dir,
        Σ_dir,
        case_name,
        FT(t_start),
        FT(t_end),
        FT(Σ_t_start),
        FT(Σ_t_end),
        z_obs,
        z_rectified_obs,
        namelist,
        y_type,
        Σ_type,
    )
end

get_t_start(m::ReferenceModel) = m.y_t_start
get_t_end(m::ReferenceModel) = m.y_t_end
get_t_start_Σ(m::ReferenceModel) = m.Σ_t_start
get_t_end_Σ(m::ReferenceModel) = m.Σ_t_end

"Returns the observed vertical locations for a reference model"
get_z_obs(m::ReferenceModel) = m.z_obs
get_z_rectified_obs(m::ReferenceModel) = m.z_rectified_obs # we need to add get_z_rectified_obs or the z_rectifier to the ReferenceModel struct, better just to add the obs

get_y_dir(m::ReferenceModel) = m.y_dir
get_Σ_dir(m::ReferenceModel) = m.Σ_dir
get_scm_namelist(m::ReferenceModel) = deepcopy(m.namelist)

# TODO: cache filename and move `get_stats_path` call to constructor.
y_nc_file(m::ReferenceModel) = get_stats_path(get_y_dir(m))
Σ_nc_file(m::ReferenceModel) = get_stats_path(get_Σ_dir(m))

data_directory(root::AbstractString, name::AbstractString, suffix::AbstractString) =
    joinpath(root, "Output.$name.$suffix")

namelist_directory(root::String, m::ReferenceModel) = namelist_directory(root, m.case_name)
namelist_directory(root::AbstractString, casename::AbstractString) = joinpath(root, "namelist_$casename.in")

num_vars(m::ReferenceModel) = length(m.y_names)

"""
    get_scm_namelist(case_name; [les_dir, overwrite, namelist_args, seed])

Returns a TurbulenceConvection.jl namelist, given a case and a list of namelist arguments.

Inputs:
 - `case_name`      :: Name of the TurbulenceConvection.jl case considered.
 - `les_dir`          :: Directory with LES data to drive the SCM with, if `case_name` is `LES_driven_SCM`.
 - `namelist_args`  :: Vector of non-default arguments to be used in the namelist, defined as a vector of tuples.
 - `seed`           :: If set, seed is an integer, and is the seed value to generate a TC namelist.
Outputs:
 - `namelist`       :: The TurbulenceConvection.jl namelist.
"""
function get_scm_namelist(
    case_name::String;
    les_dir::OptString = nothing,
    namelist_args::OptVec{<:Tuple} = nothing,
    seed::OptInt = nothing,
)::Dict
    namelist = if isnothing(seed)
        NameList.default_namelist(case_name; write = false, set_seed = false)
    else
        NameList.default_namelist(case_name; write = false, set_seed = true, seed = seed)
    end

    namelist["stats_io"]["calibrate_io"] = true

    update_namelist!(namelist, namelist_args)

    # if `LES_driven_SCM` case, provide input LES stats file
    if case_name == "LES_driven_SCM"
        @assert !isnothing(les_dir) "lesfile must be specified in the construction of LES_driven_SCM namelist."
        namelist["meta"]["lesfile"] = get_stats_path(les_dir)
    end

    return namelist
end

"""
    construct_z_obs(namelist::Dict)

Constructs the vector of observed locations given a TurbulenceConvection.jl namelist.
"""
function construct_z_obs(namelist::Dict)
    (; z_mesh) = construct_mesh(namelist)
    grid = TC.Grid(z_mesh)
    return vec(grid.zc)
end

"""
    get_ref_model_kwargs(ref_config::Dict; [global_namelist_args])

Extract fields from the reference config necessary to construct [`ReferenceModel`](@ref)s.

The namelist that defines a case is fetched from TC.jl for each case defined in `ref_config["case_name"]`.
These can be overwritten in one of two ways;
1. Define case-by-case overwrite entries in `ref_config["namelist_args"]`
2. Define global overwrite entries with the keyword argument `global_namelist_args` (`Vector` of `Tuple`s).
    These entries apply to all cases, training, validation, testing, etc.
Note that the case-by-case `namelist_args` supersede both TC.jl defaults and global `namelist_args` entries.

See also [`construct_reference_models`](@ref).
"""
function get_ref_model_kwargs(ref_config::Dict; global_namelist_args::OptVec{<:Tuple} = nothing)
    n_cases = length(ref_config["case_name"])
    Σ_dir = expand_dict_entry(ref_config, "Σ_dir", n_cases)
    Σ_t_start = expand_dict_entry(ref_config, "Σ_t_start", n_cases)
    Σ_t_end = expand_dict_entry(ref_config, "Σ_t_end", n_cases)
    n_obs = expand_dict_entry(ref_config, "n_obs", n_cases)
    y_type = ref_config["y_reference_type"]
    Σ_type = ref_config["Σ_reference_type"]
    z_rectifiers = expand_dict_entry(ref_config, "z_rectifiers", n_cases) # expand if it's not a vector, if is a vector, must be of length n_cases (so even if reusing same custom z_vector, repeat it n_cases times), if not set, will return [nothing,] x n_cases
    # Construct namelist_args from case-specific args merged with global args
    # Note: Case-specific args takes precedence over global args
    case_namelist_args = expand_dict_entry(ref_config, "namelist_args", n_cases)
    namelist_args = merge_namelist_args.(Ref(global_namelist_args), case_namelist_args)

    rm_kwargs = Dict(
        :y_names => ref_config["y_names"],
        # Reference path specification
        :y_dir => ref_config["y_dir"],
        :Σ_dir => Σ_dir,
        # Case name
        :case_name => ref_config["case_name"],
        # Define observation window (s)
        :t_start => ref_config["t_start"],
        :t_end => ref_config["t_end"],
        :Σ_t_start => Σ_t_start,
        :Σ_t_end => Σ_t_end,
        :n_obs => n_obs,
        :namelist_args => namelist_args,
        :y_type => y_type,
        :Σ_type => Σ_type,
        :z_rectifiers => z_rectifiers,
    )
    n_RM = length(rm_kwargs[:case_name])
    for (k, v) in pairs(rm_kwargs)
        if !(k in [:y_type, :Σ_type])
            @assert length(v) == n_RM "Entry `$k` in the reference config file has length $(length(v)). Should have length $n_RM."
        end
    end
    return rm_kwargs
end

"""
    construct_reference_models(kwarg_ld::Dict; [seed])::Vector{ReferenceModel}

Returns a vector of `ReferenceModel`s given a dictionary of keyword argument lists.

See also [`get_ref_model_kwargs`](@ref).

Inputs:

 - `kwarg_ld`   :: Dictionary of keyword argument lists
 - `seed`       :: If set, seed is an integer, and is the seed value to generate a TC namelist for each case

Outputs:

 - `ref_models` :: Vector where the i-th ReferenceModel is constructed from the i-th element of every keyword argument list of the dictionary.
"""
function construct_reference_models(kwarg_ld::Dict; seed::OptInt = nothing)::Vector{ReferenceModel}
    n_RM = length(kwarg_ld[:case_name])
    ref_models = Vector{ReferenceModel}()
    for RM_i in 1:n_RM
        kw = Dict(k => v[RM_i] for (k, v) in pairs(kwarg_ld) if !(k in [:y_type, :Σ_type]))  # unpack dict (ytype and Σ_type must be the same for all models so we just get them from kwarg_ld  )
        args = (kw[j] for j in (:y_names, :y_dir, :case_name, :t_start, :t_end))

        push!(
            ref_models,
            ReferenceModel(
                args...,
                Σ_dir = get(kw, :Σ_dir, nothing),
                Σ_t_start = get(kw, :Σ_t_start, nothing),
                Σ_t_end = get(kw, :Σ_t_end, nothing),
                y_type = kwarg_ld[:y_type],
                Σ_type = kwarg_ld[:Σ_type],
                n_obs = get(kw, :n_obs, nothing),
                namelist_args = get(kw, :namelist_args, nothing),
                seed = seed,
                z_rectifier = get(kw, :z_rectifiers, nothing), # function/vector/z_bounds for choosing the z you want for the comparison
            ),
        )
    end
    return ref_models
end

"""
    time_shift_reference_model(m::ReferenceModel, time_shift::FT) where {FT <: Real}

Returns a time-shifted ReferenceModel, considering an interval relative to the last
available time step of the original model. Only LES data (from y_dir or Σ_dir) are time shifted.

Inputs:

 - `m`     :: A ReferenceModel.
 - `time_shift`  :: [LES last time - SCM start time (LES timeframe)]

Outputs:

 - The time-shifted ReferenceModel.
"""
function time_shift_reference_model(m::ReferenceModel, time_shift::FT) where {FT <: Real}
    filename = y_nc_file(m)
    t_mean = nc_fetch(filename, "t")

    filename = Σ_nc_file(m)
    t_Σ = nc_fetch(filename, "t")

    t_start = isa(m.y_type, LES) ? t_mean[end] - time_shift + get_t_start(m) : get_t_start(m)
    t_end = isa(m.y_type, LES) ? t_mean[end] - time_shift + get_t_end(m) : get_t_end(m)
    Σ_t_start = isa(m.Σ_type, LES) ? t_Σ[end] - time_shift + get_t_start_Σ(m) : get_t_start_Σ(m)
    Σ_t_end = isa(m.Σ_type, LES) ? t_Σ[end] - time_shift + get_t_end_Σ(m) : get_t_end_Σ(m)

    @assert t_start >= 0 "t_start must be positive after time shift, but $t_start was given."
    @assert Σ_t_start >= 0 "Σ_t_start must be positive after time shift, but $Σ_t_start was given."
    @info string(
        "Shifting time windows for ReferenceModel $(m.case_name) ",
        "to ty=($t_start, $t_end), tΣ=($Σ_t_start, $Σ_t_end).",
    )
    return ReferenceModel(
        m.y_names,
        m.y_dir,
        m.Σ_dir,
        m.case_name,
        t_start,
        t_end,
        Σ_t_start,
        Σ_t_end,
        m.z_obs,
        m.z_rectified_obs, # we added this to the ReferenceModel struct, so the default constructor has it
        m.namelist,
        m.y_type,
        m.Σ_type,
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


    ReferenceModelBatch(kwarg_ld::Dict, shuffling::Bool = true)

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

function ReferenceModelBatch(kwarg_ld::Dict, shuffling::Bool = true)
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
