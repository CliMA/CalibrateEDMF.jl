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
include(joinpath("ReferenceStats", "runtests.jl"))
include(joinpath("Pipeline", "runtests.jl"))
include(joinpath("NetCDFIO", "runtests.jl"))

@testset "error_utils" begin
    # Vector of vectors vs vector
    foo = rand(5)
    foo_vec = [foo, foo]
    foo_vec2 = [foo .+ 1.0, foo .+ 2.0]

    @test compute_mse(foo_vec, foo) ≈ [0, 0]
    @test compute_mse(foo_vec2, foo) ≈ [1, 4]

    # Matrices vs vector
    foo = rand(5)
    bar = zeros(5, 2)
    bar2 = zeros(2, 5)
    bar3 = zeros(5, 2)
    bar4 = zeros(6, 2)
    for i in 1:size(bar, 2)
        bar[:, i] = foo
        bar2[i, :] = foo
        bar3[:, i] = foo .+ i
    end

    @test compute_mse(bar, foo) ≈ [0, 0]
    @test compute_mse(bar2, foo) ≈ [0, 0]
    @test compute_mse(bar3, foo) ≈ [1, 4]
    @test_throws BoundsError compute_mse(bar4, foo)
end
