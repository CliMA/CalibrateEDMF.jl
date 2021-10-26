using Test
using Distributions
using Random
using CalibrateEDMF
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))


include(joinpath("DistributionUtils", "runtests.jl"))
include(joinpath("LESUtils", "runtests.jl"))
include(joinpath("TurbulenceConvectionUtils", "runtests.jl"))
include(joinpath("ReferenceModels", "runtests.jl"))
include(joinpath("Pipeline", "runtests.jl"))

@testset "error_utils" begin
    foo = rand(5)
    foo_vec = [foo, foo]
    foo_vec2 = [foo .+ 1.0, foo .+ 2.0]

    @test compute_errors(foo_vec, foo) ≈ [0, 0]
    @test compute_errors(foo_vec2, foo) != [sqrt(5), 2 * sqrt(5)]
end
