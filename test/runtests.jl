using Test
using Distributions
using Random
include(joinpath(@__DIR__, "../src/helper_funcs.jl"))


@testset "ReferenceModel" begin
    les_root = "foo"
    scm_root = "bar"
    sim_name = "foo1"
    les_suffix = "bar1"
    y_names = ["thetal_mean", "ql_mean", "qt_mean"]
    t_start = 0.0
    t_end = 10.0

    ref_model = ReferenceModel(
        y_names = y_names,
        les_root = les_root,
        les_name = sim_name,
        les_suffix = les_suffix,
        scm_root = scm_root,
        scm_name = sim_name,
        t_start = t_start,
        t_end = t_end,
    )

    @test les_dir(ref_model) == joinpath(les_root, "Output.$sim_name.$les_suffix")
    @test scm_dir(ref_model) == joinpath(scm_root, "Output.$sim_name.00000")
    @test num_vars(ref_model) == 3
end


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
