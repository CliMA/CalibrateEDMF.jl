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
    # Field names for cost function
    y_names::Vector{String}
    # Directory of les simulation
    les_dir::String
    # Directory of scm simulation
    scm_dir::String
    # Name of case
    scm_name::String
    # TODO: Make t_start and t_end vectors for multiple time intervals per reference model.
    "Start time for computing statistics over"
    t_start::Real
    "End time for computing statistics over"
    t_end::Real
end

les_dir(m::ReferenceModel) = m.les_dir
scm_dir(m::ReferenceModel) = m.scm_dir
data_directory(root::S, name::S, suffix::S) where {S <: AbstractString} = joinpath(root, "Output.$name.$suffix")

namelist_directory(root::String, m::ReferenceModel) = namelist_directory(root, m.scm_name)
namelist_directory(root::S, casename::S) where {S <: AbstractString} = joinpath(root, "namelist_$casename.in")

num_vars(m::ReferenceModel) = length(m.y_names)


end # module
