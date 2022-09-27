using Test
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.TurbulenceConvectionUtils
import CalibrateEDMF.HelperFuncs: get_entry
using Random

pwdir = mktempdir()
@testset "ReferenceModel" begin
    les_dir_test = joinpath(pwdir, "foo/bar/les")
    case_name_test = "DYCOMS_RF01"
    y_names = ["thetal_mean", "ql_mean", "qt_mean"]
    ti = 0.0
    tf = 10.0

    ref_model = ReferenceModel(y_names, les_dir_test, case_name_test, ti, tf)

    @test get_y_dir(ref_model) == joinpath(pwdir, "foo/bar/les")
    @test get_Σ_dir(ref_model) == joinpath(pwdir, "foo/bar/les")
    @test num_vars(ref_model) == 3
    @test get_t_start(ref_model) == 0.0
    @test get_t_start_Σ(ref_model) == 0.0
    @test get_t_end(ref_model) == 10.0
    @test get_t_end_Σ(ref_model) == 10.0
    z_obs = get_z_obs(ref_model)
    @test isa(z_obs, Array)
    # Check monotonicity
    for (i, height) in enumerate(z_obs[1:(end - 1)])
        @test height < z_obs[i + 1]
    end

    # Test coarse model
    n_obs = 20
    ref_model_coarse = ReferenceModel(y_names, les_dir_test, case_name_test, ti, tf, n_obs = n_obs)
    z_coarse = get_z_obs(ref_model_coarse)
    @test isa(z_coarse, Array)
    @test length(z_coarse) == n_obs

    # Test stretched grid
    namelist_args = [("grid", "stretch", "flag", true)]
    ref_model_stretched = ReferenceModel(y_names, les_dir_test, case_name_test, ti, tf; namelist_args = namelist_args)
    z_stretch = get_z_obs(ref_model_stretched)
    @test isa(z_stretch, Array)
    # Test stretching
    @test z_stretch[2] - z_stretch[1] < z_stretch[3] - z_stretch[2]
    @test z_stretch[3] - z_stretch[2] < z_stretch[4] - z_stretch[3]
end

@testset "ReferenceModelBatch" begin
    y_dir_test = joinpath(pwdir, "foo/bar/scm/Output.test_case.12345")
    # Choose same SCM to speed computation
    y_dirs = repeat([y_dir_test], 2)
    kwargs_ref_model = Dict(
        :y_names => [["u_mean"], ["v_mean", "thetal_mean"]],
        :y_dir => y_dirs,
        :case_name => ["DYCOMS_RF01", "GABLS"],
        :t_start => repeat([0.0], 2),
        :t_end => repeat([10.0], 2),
        :y_type => SCM(),
        :Σ_type => SCM(),
    )
    ref_model_batch = ReferenceModelBatch(kwargs_ref_model)
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
    uuid = "01"
    case = "Bomex"
    y_dirs = repeat([joinpath(data_dir, "Output.$case.$uuid")], 2)
    # Reduce resolution and t_max to speed computation as well
    t_max = 4 * 3600.0
    namelist_args = [
        ("time_stepping", "t_max", t_max),
        ("time_stepping", "dt_max", 30.0),
        ("time_stepping", "dt_min", 20.0),
        ("stats_io", "frequency", 720.0),
        ("grid", "dz", 150.0),
        ("grid", "nz", 20),
    ]

    kwargs_ref_model = Dict(
        :y_names => [["u_mean", "v_mean"], ["thetal_mean"]],
        :y_dir => y_dirs,
        :case_name => repeat([case], 2),
        :t_start => repeat([t_max - 3600], 2),
        :t_end => [t_max, t_max],
        :Σ_t_start => repeat([t_max - 2.0 * 3600], 2),
        :Σ_t_end => repeat([t_max - 0.5 * 3600], 2),
        :n_obs => [nothing, 15],
        :namelist_args => repeat([namelist_args], 2),
        :y_type => SCM(),
        :Σ_type => SCM(),
    )
    ref_models = construct_reference_models(kwargs_ref_model)
    run_reference_SCM.(ref_models; output_root = data_dir, uuid = uuid, overwrite = true, run_single_timestep = false)
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

@testset "ReferenceModel namelist_args prioritization" begin
    config = Dict(
        "reference" => Dict(
            "case_name" => ["Bomex", "Rico", "TRMM_LBA"],
            "y_dir" => ["Output.Bomex.01", "Output.Rico.01", "Output.TRMM_LBA.01"],
            "scm_dir" => ["Output.Bomex.01", "Output.Rico.01", "Output.TRMM_LBA.01"],
            "y_names" => repeat([["thetal_mean"]], 3),
            "t_start" => [0.0, 1.0, 1.0],
            "t_end" => [1.0, 2.0, 2.0],
            "y_reference_type" => SCM(),
            "Σ_reference_type" => SCM(),
            "namelist_args" => [
                [("time_stepping", "dt_max", 21.1), ("stats_io", "frequency", 121.0)],
                [("time_stepping", "dt_max", 29.1), ("stats_io", "frequency", 122.0)],
                nothing,
            ],
        ),
        "scm" => Dict("namelist_args" => [("time_stepping", "dt_max", 30.1), ("grid", "nz", 23)]),
    )
    # 1a. Test that overwriting works correctly: i.e. case-specific namelist args overwrite global (scm-dict) namelist args
    # 1b. Test that `nothing` in case-specific namelist args works as expected
    ref_config = config["reference"]
    namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    kwargs_ref_model = get_ref_model_kwargs(ref_config; global_namelist_args = namelist_args)
    ref_models = construct_reference_models(kwargs_ref_model)

    # 1a. Overwriting works correctly
    @test ref_models[1].namelist["time_stepping"]["dt_max"] == 21.1
    @test ref_models[1].namelist["stats_io"]["frequency"] == 121.0
    @test ref_models[1].namelist["grid"]["nz"] == 23
    @test ref_models[2].namelist["time_stepping"]["dt_max"] == 29.1
    @test ref_models[2].namelist["stats_io"]["frequency"] == 122.0
    @test ref_models[2].namelist["grid"]["nz"] == 23
    # 1b. empty case-specific namelist args handled correctly
    @test ref_models[3].namelist["time_stepping"]["dt_max"] == 30.1
    @test ref_models[3].namelist["grid"]["nz"] == 23

    # 2. Check that `nothing` global namelist args handled correctly
    kwargs_ref_model = get_ref_model_kwargs(ref_config; global_namelist_args = nothing)
    ref_models = construct_reference_models(kwargs_ref_model)

    @test ref_models[1].namelist["time_stepping"]["dt_max"] == 21.1
    @test ref_models[2].namelist["time_stepping"]["dt_max"] == 29.1
    @test ref_models[3].namelist["time_stepping"]["dt_max"] != 30.1

    # 3. Check that no passed case-specific, nor global namelist arguments works correctly
    pop!(ref_config, "namelist_args")
    kwargs_ref_model = get_ref_model_kwargs(ref_config; global_namelist_args = nothing)
    ref_models = construct_reference_models(kwargs_ref_model)

    @test ref_models[1].namelist["time_stepping"]["dt_max"] != 21.1
    @test ref_models[2].namelist["time_stepping"]["dt_max"] != 29.1
    @test ref_models[3].namelist["time_stepping"]["dt_max"] != 30.1
end
