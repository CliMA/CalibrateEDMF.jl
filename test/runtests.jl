using Test
using Distributions
using Random
using CalibrateEDMF
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))


include(joinpath("DistributionUtils", "runtests.jl"))
include(joinpath("ReferenceModels", "runtests.jl"))

@testset "les_handlers" begin
    scm_names = ["total_flux_qt", "total_flux_h", "u_mean"]
    @test get_les_names(scm_names, "GABLS") == ["resolved_z_flux_qt", "resolved_z_flux_theta", "u_translational_mean"]

    @test get_les_names(scm_names, "foo") == ["resolved_z_flux_qt", "resolved_z_flux_thetali", "u_translational_mean"]
end


@testset "error_utils" begin
    foo = rand(5)
    foo_vec = [foo, foo]
    foo_vec2 = [foo .+ 1.0, foo .+ 2.0]

    @test compute_errors(foo_vec, foo) ≈ [0, 0]
    @test compute_errors(foo_vec2, foo) != [sqrt(5), 2 * sqrt(5)]
end

@testset "general_utils" begin
    @test generate_uuid(1, forcing_model = "model1", month = 1, experiment = "experiment1") == "1_model1_01_experiment1"
end
