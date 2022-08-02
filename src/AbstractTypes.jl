module AbstractTypes

export OptVec, OptString, OptInt, OptReal, Opt

# abstract types

"Optional argument"
const Opt{T} = Union{Nothing, T}

"Optional vector"
const OptVec{T} = Union{Nothing, Vector{T}}

"Optional string"
const OptString = Union{String, Nothing}

"Optional integer"
const OptInt = Union{Integer, Nothing}

"Optional real"
const OptReal = Union{Real, Nothing}

end # module
