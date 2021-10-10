module ReferenceModels

export ReferenceModel
export les_dir, scm_dir, num_vars
export data_directory, namelist_directory


"""
    struct ReferenceModel
    
A structure containing information about the 'true' reference model
and the observation map used to compare the parameterized and
reference models.
"""
Base.@kwdef struct ReferenceModel
    "Vector of reference variable names"
    y_names::Vector{String}

    "Root directory for reference LES data"
    les_root::String
    "Name of LES reference simulation file"
    les_name::String
    "Suffix of LES reference simulation file"
    les_suffix::String

    "Root directory for SCM data (used for interpolation)"
    scm_root::String
    "Name of SCM reference simulation file"
    scm_name::String
    "Suffix of SCM reference simulation file"
    scm_suffix::String = "00000"

    # TODO: Make t_start and t_end vectors for multiple time intervals per reference model.
    "Start time for computing statistics over"
    t_start::Real
    "End time for computing statistics over"
    t_end::Real
end

les_dir(m::ReferenceModel) = data_directory(m.les_root, m.les_name, m.les_suffix)
scm_dir(m::ReferenceModel) = data_directory(m.scm_root, m.scm_name, m.scm_suffix)
data_directory(root::S, name::S, suffix::S) where {S <: AbstractString} = joinpath(root, "Output.$name.$suffix")

namelist_directory(root::String, m::ReferenceModel) = namelist_directory(root, m.scm_name)
namelist_directory(root::S, casename::S) where {S <: AbstractString} = joinpath(root, "namelist_$casename.in")

num_vars(m::ReferenceModel) = length(m.y_names)


end # module
