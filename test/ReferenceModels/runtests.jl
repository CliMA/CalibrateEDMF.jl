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
    @test_throws Exception get_z_obs(ref_model)
end

@testset "ReferenceModelBatch" begin
    scm_dir_test = "/foo/bar/scm/Output.test_case.12345"
    # Choose same SCM to speed computation
    scm_dirs = repeat([scm_dir_test], 2)
    kwargs_ref_model = Dict(
        :y_names => [["u_mean"], ["v_mean", "thetal_mean"]],
        :y_dir => scm_dirs,
        :scm_dir => scm_dirs,
        :case_name => ["foo", "bar"],
        :t_start => repeat([0.0], 2),
        :t_end => repeat([10.0], 2),
    )
    ref_model_batch = construct_ref_model_batch(kwargs_ref_model)
    @test length(ref_model_batch.ref_models) == 2
    @test length(ref_model_batch.eval_order) == 2

    # Get a minibatch and check that the eval order shrinks
    ref_models, model_indices = get_minibatch!(ref_model_batch, 1)

    @test isa(ref_models, Vector{ReferenceModel})
    @test isa(model_indices, Vector{Int})
    @test length(ref_model_batch.ref_models) == 2
    @test length(ref_model_batch.eval_order) == 1
end

@testset "ReferenceModel handlers" begin
    # Choose same SCM to speed computation
    data_dir = mktempdir()
    scm_dirs = repeat([joinpath(data_dir, "Output.Bomex.000000")], 2)
    # Reduce resolution and t_max to speed computation as well
    t_max = 4 * 3600.0
    namelist_args = [
        ("time_stepping", "t_max", t_max),
        ("time_stepping", "dt_max", 30.0),
        ("time_stepping", "dt_min", 20.0),
        ("grid", "dz", 150.0),
        ("grid", "nz", 20),
        ("stats_io", "frequency", 720.0),
    ]
    kwargs_ref_model = Dict(
        :y_names => [["u_mean", "v_mean"], ["thetal_mean"]],
        :y_dir => scm_dirs,
        :scm_dir => scm_dirs,
        :case_name => repeat(["Bomex"], 2),
        :t_start => repeat([t_max - 3600], 2),
        :t_end => [t_max, t_max],
        :Σ_t_start => repeat([t_max - 2.0 * 3600], 2),
        :Σ_t_end => repeat([t_max - 0.5 * 3600], 2),
        :n_obs => [nothing, 15],
    )
    ref_models = construct_reference_models(kwargs_ref_model)
    run_reference_SCM.(ref_models, overwrite = false, run_single_timestep = false, namelist_args = namelist_args)
    Δt = 3.0 * 3600
    ref_model = time_shift_reference_model(ref_models[1], Δt)
    z = get_z_obs(ref_model)

    @test get_t_start(ref_model) == t_max - Δt + (t_max - 3600)
    @test get_t_end(ref_model) == t_max - Δt + t_max
    @test get_t_start_Σ(ref_model) == t_max - Δt + (t_max - 2.0 * 3600)
    @test get_t_end_Σ(ref_model) == t_max - Δt + (t_max - 0.5 * 3600)
    @test length(z) == 20
    @test z[2] - z[1] ≈ 150

    Δt = 2 * t_max
    @test_throws AssertionError time_shift_reference_model(ref_models[1], Δt)

    ref_model = ref_models[2]
    z = get_z_obs(ref_model)

    @test get_t_start(ref_model) == t_max - 3600
    @test get_t_end(ref_model) == t_max
    @test get_t_start_Σ(ref_model) == t_max - 2.0 * 3600
    @test get_t_end_Σ(ref_model) == t_max - 0.5 * 3600
    @test length(z) == 15
    @test z[2] - z[1] ≈ 150.0 * (20 - 1) / (15 - 1)

end
