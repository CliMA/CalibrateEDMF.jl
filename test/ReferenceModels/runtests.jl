using Test
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.TurbulenceConvectionUtils

@testset "ReferenceModel" begin
    les_dir_test = "/foo/bar/les"
    scm_dir_test = "/foo/bar/scm/Output.test_case.12345"
    case_name_test = "test_case"
    y_names = ["thetal_mean", "ql_mean", "qt_mean"]
    ti = 0.0
    tf = 10.0

    ref_model = ReferenceModel(y_names, les_dir_test, scm_dir_test, case_name_test, ti, tf)

    @test y_dir(ref_model) == "/foo/bar/les"
    @test Σ_dir(ref_model) == "/foo/bar/les"
    @test scm_dir(ref_model) == "/foo/bar/scm/Output.test_case.12345"
    @test num_vars(ref_model) == 3
    @test uuid(ref_model) == "12345"
    @test get_t_start(ref_model) == 0.0
    @test get_t_start_Σ(ref_model) == 0.0
    @test get_t_end(ref_model) == 10.0
    @test get_t_end_Σ(ref_model) == 10.0
end

@testset "ReferenceModel handlers" begin
    # Choose same SCM to speed computation
    data_dir = mktempdir()
    scm_dirs = repeat([joinpath(data_dir, "Output.Bomex.000000")], 2)
    kwargs_ref_model = Dict(
        :y_names => [["u_mean"], ["v_mean"]],
        :y_dir => scm_dirs,
        :scm_dir => scm_dirs,
        :case_name => repeat(["Bomex"], 2),
        :t_start => [4.0 * 3600, 4.0 * 3600],
        :t_end => [6.0 * 3600, 6.0 * 3600],
    )
    ref_models = construct_reference_models(kwargs_ref_model)
    run_SCM(ref_models, overwrite = false)
    Δt_y = 3600.0
    Δt_Σ = 5400.0
    ref_model = time_shift_reference_model(ref_models[1], Δt_y, Δt_Σ)

    @test get_t_start(ref_model) == 21600.0 - Δt_y
    @test get_t_start_Σ(ref_model) == 21600.0 - Δt_Σ
    @test get_t_end(ref_model) == get_t_end_Σ(ref_model)
    @test get_t_end(ref_model) == 21600.0

    Δt_y = 2 * 21600.0
    Δt_Σ = 2 * 21600.0
    @test_throws AssertionError time_shift_reference_model(ref_models[2], Δt_y, Δt_Σ)


end
