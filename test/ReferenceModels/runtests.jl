using Test
using CalibrateEDMF.ReferenceModels


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
