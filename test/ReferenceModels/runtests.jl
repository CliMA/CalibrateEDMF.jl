using Test
using CalibrateEDMF.ReferenceModels


@testset "ReferenceModel" begin
    les_dir_test = "/foo/bar/les"
    scm_dir_test = "/foo/bar/scm/Output.test_case.12345"
    case_name_test = "test_case"
    y_names = ["thetal_mean", "ql_mean", "qt_mean"]
    t_start = 0.0
    t_end = 10.0

    ref_model = ReferenceModel(
        y_names = y_names,
        les_dir = les_dir_test,
        scm_dir = scm_dir_test,
        case_name = case_name_test,
        t_start = t_start,
        t_end = t_end,
    )

    @test les_dir(ref_model) == ref_model.les_dir
    @test scm_dir(ref_model) == ref_model.scm_dir
    @test les_dir(ref_model) == "/foo/bar/les"
    @test scm_dir(ref_model) == "/foo/bar/scm/Output.test_case.12345"
    @test num_vars(ref_model) == 3
    @test uuid(ref_model) == "12345"
end
