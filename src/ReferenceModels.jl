module ReferenceModels

export ReferenceModel
export les_dir, scm_dir, num_vars, uuid
export data_directory, namelist_directory
export construct_reference_models

"""
    struct ReferenceModel
    
A structure containing information about the 'true' reference model
and the observation map used to compare the parameterized and
reference models.
"""
Base.@kwdef struct ReferenceModel
    "Vector of reference variable names"
    # Field names for cost function
    y_names::Vector{String}
    # Directory of les simulation
    les_dir::String
    # Directory of scm simulation
    scm_dir::String
    # Name of case
    case_name::String
    # TODO: Make t_start and t_end vectors for multiple time intervals per reference model.
    "Start time for computing statistics over"
    t_start::Real
    "End time for computing statistics over"
    t_end::Real
end
function ReferenceModel(
    case_name::String;
    scm_parent_dir::String = "scm_init",
    scm_suffix::String = "00000",
    kwargs...,
) where {FT <: Real}
    scm_dir = data_directory(scm_parent_dir, case_name, scm_suffix)
    return ReferenceModel(; case_name = case_name, scm_dir = scm_dir, kwargs...)
end

les_dir(m::ReferenceModel) = m.les_dir
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
 - ref_models :: Vector where the i-th ReferenceModel is constructed from the i-th element
    of every keyword argument list of the dictionary.
"""
function construct_reference_models(kwarg_ld::Dict{Symbol, Vector{T} where T})::Vector{ReferenceModel}
    num_ref_models = length(collect(values(kwarg_ld))[1])
    ref_models = Vector{ReferenceModel}()
    for ref_model_id in range(1, stop = num_ref_models)
        ref_model_kwargs = Dict()
        for (key, value) in pairs(kwarg_ld)
            ref_model_kwargs[key] = value[ref_model_id]
        end
        push!(ref_models, ReferenceModel(; ref_model_kwargs...))
    end
    return ref_models
end

function construct_reference_models(
    case_names::Vector{String},
    kwarg_ld::Dict{Symbol, Vector{T} where T},
)::Vector{ReferenceModel}
    num_ref_models = length(collect(values(kwarg_ld))[1])
    ref_models = Vector{ReferenceModel}()
    for ref_model_id in range(1, stop = num_ref_models)
        ref_model_kwargs = Dict()
        for (key, value) in pairs(kwarg_ld)
            ref_model_kwargs[key] = value[ref_model_id]
        end
        push!(ref_models, ReferenceModel(case_names[ref_model_id]; ref_model_kwargs...))
    end
    return ref_models
end

end # module
