using JSON
using Test
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.TurbulenceConvectionUtils


@testset "TurbulenceConvectionUtils" begin
    @test get_gcm_les_uuid(1, forcing_model = "model1", month = 1, experiment = "experiment1") ==
          "1_model1_01_experiment1"
end

@testset "TC.jl error handling" begin
    # Choose same SCM to speed computation
    data_dir = mktempdir()
    scm_dirs = [joinpath(data_dir, "Output.Bomex.000000")]
    # Violate CFL condition for TC.jl simulation to fail
    t_max = 2 * 3600.0
    namelist_args = [
        ("time_stepping", "t_max", t_max),
        ("time_stepping", "dt_max", 200.0),
        ("time_stepping", "dt_min", 200.0),
        ("grid", "dz", 150.0),
        ("grid", "nz", 20),
        ("stats_io", "frequency", 720.0),
    ]
    kwargs_ref_model = Dict(
        :y_names => [["u_mean", "v_mean"]],
        :y_dir => scm_dirs,
        :scm_dir => scm_dirs,
        :case_name => ["Bomex"],
        :t_start => [t_max - 3600],
        :t_end => [t_max],
        :Σ_t_start => [t_max - 2.0 * 3600],
        :Σ_t_end => [t_max],
    )
    ref_models = construct_reference_models(kwargs_ref_model)
    run_reference_SCM.(ref_models, run_single_timestep = false, namelist_args = namelist_args)

    u = [0.15]
    u_names = ["entrainment_factor"]
    res_dir, model_error = run_SCM_handler(ref_models[1], data_dir, u, u_names, namelist_args)

    @test model_error
end

@testset "Namelist modification" begin

    namelist_compare_entries = ["microphysics", "time_stepping", "stats_io", "turbulence", "grid", "thermodynamics"]
    # Choose same SCM to speed computation
    data_dir = mktempdir()
    scm_dirs = [joinpath(data_dir, "Output.Bomex.000000")]
    case_name = "Bomex"
    t_max = 2 * 3600.0

    kwargs_ref_model = Dict(
        :y_names => [["u_mean", "v_mean"]],
        :y_dir => scm_dirs,
        :scm_dir => scm_dirs,
        :case_name => [case_name],
        :t_start => [t_max - 3600],
        :t_end => [t_max],
        :Σ_t_start => [t_max - 2.0 * 3600],
        :Σ_t_end => [t_max],
    )
    ref_models = construct_reference_models(kwargs_ref_model)
    run_reference_SCM.(ref_models, run_single_timestep = true)

    # ensure namelist generated with `run_reference_SCM` matches default namelist
    init_namelist_path = namelist_directory(scm_dir(ref_models[1]), ref_models[1])
    default_namelist = TurbulenceConvectionUtils.NameList.default_namelist(case_name, root = scm_dir(ref_models[1]))
    reference_namelist = JSON.parsefile(init_namelist_path)

    for entry in namelist_compare_entries
        @test default_namelist[entry] == reference_namelist[entry]
    end

    # ensure namelist in a `run_SCM_handler` call is modified as expected
    u = [0.15, 0.52]
    u_names = ["entrainment_factor", "detrainment_factor"]
    # Test namelist modification for different nesting levels
    namelist_args = (
        ("thermodynamics", "quadrature_type", "gaussian"),
        ("grid", "dz", 150.0),
        ("grid", "nz", 20),
        ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_adv_coeff", 0.0),
        ("turbulence", "EDMF_PrognosticTKE", "stochastic", "detr_lognormal_var", 0.2),
    )

    res_dir, model_error = run_SCM_handler(ref_models[1], data_dir, u, u_names, namelist_args)

    run_scm_namelist_path = namelist_directory(res_dir, ref_models[1])
    run_scm_namelist = JSON.parsefile(run_scm_namelist_path)
    expected_run_scm_namelist = deepcopy(default_namelist)
    expected_run_scm_namelist["turbulence"]["EDMF_PrognosticTKE"]["entrainment_factor"] = 0.15
    expected_run_scm_namelist["turbulence"]["EDMF_PrognosticTKE"]["detrainment_factor"] = 0.52
    expected_run_scm_namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_adv_coeff"] = 0.0
    expected_run_scm_namelist["turbulence"]["EDMF_PrognosticTKE"]["stochastic"]["detr_lognormal_var"] = 0.2
    expected_run_scm_namelist["thermodynamics"]["quadrature_type"] = "gaussian"
    expected_run_scm_namelist["grid"]["nz"] = 20
    expected_run_scm_namelist["grid"]["dz"] = 150.0

    for entry in namelist_compare_entries
        @test expected_run_scm_namelist[entry] == run_scm_namelist[entry]
    end
end
