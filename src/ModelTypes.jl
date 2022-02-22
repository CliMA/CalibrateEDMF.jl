module ModelTypes

abstract type ModelType end
struct LES <: ModelType end
struct SCM <: ModelType end

end
