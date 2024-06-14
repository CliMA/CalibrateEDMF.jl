using JSON
using Test

using Statistics
import StaticArrays
using NCDatasets
const SA = StaticArrays

using EnsembleKalmanProcesses.ParameterDistributions

using CalibrateEDMF.ModelTypes
using CalibrateEDMF.ReferenceModels
import CalibrateEDMF.ReferenceModels: NameList
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.TurbulenceConvectionUtils
import CalibrateEDMF.TurbulenceConvectionUtils: create_parameter_vectors
import CalibrateEDMF.HelperFuncs: do_nothing_param_map, change_entry!, update_namelist!

@testset "TurbulenceConvectionUtils" begin
    @testset "test create_parameter_vectors" begin

        namelist = Dict(
            "turbulence" => Dict("EDMF_PrognosticTKE" => Dict("foo" => -1.0)),
            "microphysics" => Dict("bar" => [-2.0, -3.0], "baz" => -4.0),
        )

        # only scalar parameters
        u_names = ["foo", "baz"]
        u = [1.0, 2.0]
        param_map = do_nothing_param_map()
        u_names_out, u_out = create_parameter_vectors(u_names, u, param_map, namelist)
        @test all(u_names_out .∈ [["foo", "baz"]])
        @test only(u_out[u_names_out .== "foo"]) == 1.0
        @test only(u_out[u_names_out .== "baz"]) == 2.0

        # scalar and vector parameters, unsorted
        u_names = ["bar_{2}", "foo", "bar_{1}"]
        u = [1.0, 2.0, 3.0]
        param_map = do_nothing_param_map()
        u_names_out, u_out = create_parameter_vectors(u_names, u, param_map, namelist)
        @test all(u_names_out .∈ [["foo", "bar"]])
        @test only(u_out[u_names_out .== "foo"]) == 2.0
        @test only(u_out[u_names_out .== "bar"]) == [3.0, 1.0]

        # scalars and larger vector parameter, sorted
        vect_names = ["bar_{$(i)}" for i in 1:20]
        u_names = ["foo", vect_names..., "baz"]
        u = randn(length(u_names))
        param_map = do_nothing_param_map()
        u_names_out, u_out = create_parameter_vectors(u_names, u, param_map, namelist)
        @test all(u_names_out .∈ [["foo", "bar", "baz"]])
        @test only(u_out[u_names_out .== "foo"]) == u[1]
        @test only(u_out[u_names_out .== "baz"]) == u[end]
        @test only(u_out[u_names_out .== "bar"]) == u[2:21]

        # scalars and larger vector parameter, decreasing order
        # check that parameter vectors components are in increasing order (sorted by vector component index)
        vect_names = ["bar_{$(i)}" for i in 22:-1:1]
        u_names = ["foo", vect_names..., "baz"]
        u = randn(length(u_names))
        param_map = do_nothing_param_map()
        u_names_out, u_out = create_parameter_vectors(u_names, u, param_map, namelist)
        @test all(u_names_out .∈ [["foo", "bar", "baz"]])
        @test only(u_out[u_names_out .== "foo"]) == u[1]
        @test only(u_out[u_names_out .== "baz"]) == u[end]
        @test only(u_out[u_names_out .== "bar"]) == u[23:-1:2]
    end

    @test get_gcm_les_uuid(1, forcing_model = "model1", month = 1, experiment = "experiment1") ==
          "1_model1_01_experiment1"

    @testset "TC.jl error handling" begin
        # Choose same SCM to speed computation
        data_dir = mktempdir()
        case = "Rico"
        uuid = "01"
        y_dirs = [joinpath(data_dir, "Output.$case.$uuid")]
        num_linear_params = 12
        linear_ent_params = 1e2*(rand(num_linear_params) .- 0.5)
        # Violate CFL condition for TC.jl simulation to fail
        t_max = 2 * 3600.0
        namelist_args = [
            ("time_stepping", "t_max", t_max),
            ("time_stepping", "dt_max", 200.0),
            ("time_stepping", "dt_min", 200.0),
            ("grid", "dz", 150.0),
            ("grid", "nz", 20),
            ("stats_io", "frequency", 720.0),
            ("logging", "truncate_stack_trace", true),
            ("turbulence", "EDMF_PrognosticTKE", "entrainment", "None"),
            ("turbulence", "EDMF_PrognosticTKE", "ml_entrainment", "Linear"),
            ("turbulence", "EDMF_PrognosticTKE", "linear_ent_params", zeros(num_linear_params)),
            ("turbulence", "EDMF_PrognosticTKE", "linear_ent_biases", false),
        ]

        kwargs_ref_model = Dict(
            :y_names => [["u_mean", "v_mean"]],
            :y_dir => y_dirs,
            :case_name => [case],
            :t_start => [t_max - 3600],
            :t_end => [t_max],
            :Σ_t_start => [t_max - 2.0 * 3600],
            :Σ_t_end => [t_max],
            :namelist_args => [namelist_args],
            :y_type => SCM(),
            :Σ_type => SCM(),
        )
        ref_models = construct_reference_models(kwargs_ref_model)
        @test_logs (:warn,) match_mode = :any run_reference_SCM.(
            ref_models;
            output_root = data_dir,
            uuid = uuid,
            run_single_timestep = false,
        )

        constraints = Dict(
            "linear_ent_params" => [repeat([no_constraint()], num_linear_params)...],
            "dt_max" => [bounded(200.0, 300.0)],
            "τ_acnv_rai" => [no_constraint()],
        )
        prior_μ = Dict(
            "linear_ent_params" => linear_ent_params,
            "dt_max" => [210.0],
            "τ_acnv_rai" => [2500.0],
        )
        param_map = do_nothing_param_map()
        prior = construct_priors(constraints; prior_mean = prior_μ, to_file = false)
        ref_stats = ReferenceStatistics(ref_models)

        u_names, u = flatten_config_dict(prior_μ)
        u = [i[1] for i in u]

        res_dir, model_error = run_SCM_handler(ref_models[1], data_dir, u, u_names, param_map)
        @test model_error

        @test_logs (:warn,) (:error,) match_mode = :any precondition(
            u,
            prior,
            param_map,
            ref_models,
            ref_stats;
            max_counter = 1,
        )
        # test failed parameter writing and check precondition behavior

        for i = 1:3
            model_evaluator, param_failed = precondition(
            u,
            prior,
            param_map,
            ref_models,
            ref_stats;
            max_counter = i,
        )
            write_failed_parameters(param_failed, failed_path)
        end

        failed_params, param_names = Dataset(failed_path, "r") do ds
            param_values_data = ds["phi"][:,:]
            param_names = ds["params_name"][:]
            return param_values_data, param_names
        end
        @test size(failed_params) == (14,6)
        @test param_names == u_names

        failed_params_mean = mean(failed_params, dims=2)
        @test isapprox(u, failed_params_mean, rtol = 1e-1)

    end

    @testset "Namelist modification" begin
        seed = 1234
        # Choose same SCM to speed computation
        data_dir = mktempdir()
        case_name = "Bomex"
        uuid = "01"
        y_dirs = [joinpath(data_dir, "Output.$case_name.$uuid")]
        t_max = 2 * 3600.0

        kwargs_ref_model = Dict(
            :y_names => [["u_mean", "v_mean"]],
            :y_dir => y_dirs,
            :case_name => [case_name],
            :t_start => [t_max - 3600],
            :t_end => [t_max],
            :Σ_t_start => [t_max - 2.0 * 3600],
            :Σ_t_end => [t_max],
            :y_type => SCM(),
            :Σ_type => SCM(),
        )
        ref_models = construct_reference_models(kwargs_ref_model; seed = seed)
        run_reference_SCM.(ref_models; output_root = data_dir, uuid = uuid, run_single_timestep = true)

        ref_model1 = ref_models[1]
        # ensure namelist generated with `run_reference_SCM` matches default namelist
        default_namelist = NameList.default_namelist(case_name; write = false, set_seed = true, seed = seed)
        reference_namelist = get_scm_namelist(ref_model1)

        default_namelist["stats_io"]["calibrate_io"] = true  # `get_scm_namelist` (in `construct_reference_models`) sets this entry to true

        namelist_compare_entries = ["microphysics", "time_stepping", "stats_io", "grid", "thermodynamics"]
        for entry in namelist_compare_entries
            @test default_namelist[entry] == reference_namelist[entry]
        end
        # Test nested dictionary of closures with type instabilities from JSON parsing
        for (key, value) in default_namelist["turbulence"]["EDMF_PrognosticTKE"]
            @test Tuple(reference_namelist["turbulence"]["EDMF_PrognosticTKE"][key]) == Tuple(value)
        end

        # ensure namelist in a `run_SCM_handler` call is modified as expected
        u = [0.15, 0.52]
        u_names = ["entrainment_factor", "detrainment_factor"]
        param_map = do_nothing_param_map()
        # Test namelist modification for different nesting levels
        namelist_args = [
            ("thermodynamics", "quadrature_type", "gaussian"),
            ("grid", "dz", 150.0),
            ("grid", "nz", 20),
            ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_adv_coeff", 0.0),
            ("turbulence", "EDMF_PrognosticTKE", "general_stochastic_ent_params", SA.SVector(0.2, 0.2, 0.01, 0.02)),
        ]
        # Set optional namelist args
        update_namelist!(ref_model1.namelist, namelist_args)

        res_dir, model_error = run_SCM_handler(ref_model1, data_dir, u, u_names, param_map)

        run_scm_namelist_path = namelist_directory(res_dir, ref_model1)
        run_scm_namelist = JSON.parsefile(run_scm_namelist_path)
        expected_run_scm_namelist = deepcopy(default_namelist)
        expected_run_scm_namelist["turbulence"]["EDMF_PrognosticTKE"]["entrainment_factor"] = 0.15
        expected_run_scm_namelist["turbulence"]["EDMF_PrognosticTKE"]["detrainment_factor"] = 0.52
        expected_run_scm_namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_adv_coeff"] = 0.0
        expected_run_scm_namelist["turbulence"]["EDMF_PrognosticTKE"]["general_stochastic_ent_params"] =
            SA.SVector(0.2, 0.2, 0.01, 0.02)
        expected_run_scm_namelist["thermodynamics"]["quadrature_type"] = "gaussian"
        expected_run_scm_namelist["grid"]["nz"] = 20
        expected_run_scm_namelist["grid"]["dz"] = 150.0
        expected_run_scm_namelist["stats_io"]["calibrate_io"] = true

        for entry in namelist_compare_entries
            @test expected_run_scm_namelist[entry] == run_scm_namelist[entry]
        end
        # Test nested dictionary of closures with type instabilities from JSON parsing
        for (key, value) in expected_run_scm_namelist["turbulence"]["EDMF_PrognosticTKE"]
            @test Tuple(run_scm_namelist["turbulence"]["EDMF_PrognosticTKE"][key]) == Tuple(value)
        end
    end  # end @testset "Namelist modification"
end  # end @testset "TurbulenceConvectionUtils"
