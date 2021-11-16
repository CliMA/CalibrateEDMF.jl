module ReferenceModels

using NCDatasets
include("helper_funcs.jl")

export ReferenceModel
export get_t_start, get_t_end, get_t_start_Σ, get_t_end_Σ
export y_dir, Σ_dir, scm_dir, num_vars, uuid
export data_directory, namelist_directory
export construct_reference_models, time_shift_reference_model

"""
    struct ReferenceModel
    
A structure containing information about the 'true' reference model
and the observation map used to compare the parameterized and
reference models.
"""
Base.@kwdef struct ReferenceModel
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
    y_t_start::Real
    "End time for computing mean statistics over"
    y_t_end::Real
    "Start time for computing covariance statistics over"
    Σ_t_start::Real
    "End time for computing covariance statistics over"
    Σ_t_end::Real
end  # ReferenceModel struct

"""
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
    )

Construct `ReferenceModel` allowing for any or all of
`Σ_dir`, `Σ_t_start`, `Σ_t_end` to be unspecified, in
which case they take their values from
`y_dir`, `t_start` and `t_end`, respectively.
"""
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
)
    Σ_dir = isnothing(Σ_dir) ? y_dir : Σ_dir
    Σ_t_start = isnothing(Σ_t_start) ? t_start : Σ_t_start
    Σ_t_end = isnothing(Σ_t_end) ? t_end : Σ_t_end

    ReferenceModel(y_names, y_dir, Σ_dir, scm_dir, case_name, t_start, t_end, Σ_t_start, Σ_t_end)
end

"""
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
    )

Constructor using `scm_parent_dir`, `case_name`, `scm_suffix`, to define `scm_dir`.

"""
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
)
    scm_dir = data_directory(scm_parent_dir, case_name, scm_suffix)
    args = (y_names, y_dir, scm_dir, case_name, t_start, t_end)
    return ReferenceModel(args..., Σ_dir = Σ_dir, Σ_t_start = Σ_t_start, Σ_t_end = Σ_t_end)
end

get_t_start(m::ReferenceModel) = m.y_t_start
get_t_end(m::ReferenceModel) = m.y_t_end
get_t_start_Σ(m::ReferenceModel) = m.Σ_t_start
get_t_end_Σ(m::ReferenceModel) = m.Σ_t_end

y_dir(m::ReferenceModel) = m.y_dir
Σ_dir(m::ReferenceModel) = m.Σ_dir
scm_dir(m::ReferenceModel) = m.scm_dir
data_directory(root::S, name::S, suffix::S) where {S <: AbstractString} = joinpath(root, "Output.$name.$suffix")
uuid(m::ReferenceModel) = String(split(scm_dir(m), ".")[end])

namelist_directory(root::String, m::ReferenceModel) = namelist_directory(root, m.case_name)
namelist_directory(root::S, casename::S) where {S <: AbstractString} = joinpath(root, "namelist_$casename.in")

num_vars(m::ReferenceModel) = length(m.y_names)

"""
    construct_reference_models(kwarg_ld::Dict{Symbol, Vector{T} where T})::Vector{ReferenceModel}

Returns a vector of `ReferenceModel`s given a dictionary of keyword argument lists.

Inputs:
 - kwarg_ld     :: Dictionary of keyword argument lists
Outputs:
 - ref_models   :: Vector where the i-th ReferenceModel is constructed from the i-th element
    of every keyword argument list of the dictionary.
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
            throw(ArgumentError(
                "You need to specify either `scm_dir` or all of " *
                "(`scm_parent_dir`, `case_name`, `scm_suffix`) to construct a `ReferenceModel`",
            ))
        end
        push!(
            ref_models,
            ReferenceModel(
                args...,
                Σ_dir = get(kw, :Σ_dir, nothing),
                Σ_t_start = get(kw, :Σ_t_start, nothing),
                Σ_t_end = get(kw, :Σ_t_end, nothing),
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
 - m     :: A ReferenceModel.
 - Δt  :: [LES last time - SCM start time (LES timeframe)]
Outputs:
 - The time-shifted ReferenceModel.
"""
function time_shift_reference_model(m::ReferenceModel, Δt::FT) where {FT <: Real}
    t = nc_fetch(y_dir(m), "t")
    t_end = t[end] - Δt + get_t_end(m)
    Σ_t_end = t[end] - Δt + get_t_end_Σ(m)
    t_start = t[end] - Δt + get_t_start(m)
    Σ_t_start = t[end] - Δt + get_t_start_Σ(m)

    @assert t_start >= 0 "t_start must be positive after time shift, but $t_start was given."
    @assert Σ_t_start >= 0 "Σ_t_start must be positive after time shift, but $Σ_t_start was given."
    @info string(
        "Shifting time windows for ReferenceModel $(m.case_name)",
        "to ty=($t_start, $t_end), tΣ=($Σ_t_start, $t_end).",
    )

    return ReferenceModel(
        m.y_names,
        y_dir(m),
        scm_dir(m),
        m.case_name,
        t_start,
        t_end;
        Σ_dir = Σ_dir(m),
        Σ_t_start = Σ_t_start,
        Σ_t_end = Σ_t_end,
    )
end


end # module
