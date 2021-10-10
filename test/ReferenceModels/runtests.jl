using Test
using CalibrateEDMF.ReferenceModels


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
