module AbstractTypes

export OptVec

# abstract types
const OptVec{T} = Union{Nothing, Vector{T}}

end # module
