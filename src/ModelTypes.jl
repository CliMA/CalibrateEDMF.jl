module ModelTypes
export ModelType, LES, SCM

abstract type ModelType end
struct LES <: ModelType end
struct SCM <: ModelType end

end
