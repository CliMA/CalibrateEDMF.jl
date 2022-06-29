using Test
using Distributions
using Random
using CalibrateEDMF
using CalibrateEDMF.HelperFuncs
const src_dir = dirname(pathof(CalibrateEDMF))

@testset "error_utils" begin
    # Vector of vectors vs vector
    foo = rand(5)
    foo_vec = [foo, foo]
    foo_vec2 = [foo .+ 1.0, foo .+ 2.0]

    @test compute_mse(foo_vec, foo) ≈ [0, 0]
    @test compute_mse(foo_vec2, foo) ≈ [1, 4]

    # Matrices vs vector
    foo = rand(5)
    bar = zeros(5, 2)
    bar2 = zeros(2, 5)
    bar3 = zeros(5, 2)
    bar4 = zeros(6, 2)
    for i in 1:size(bar, 2)
        bar[:, i] = foo
        bar2[i, :] = foo
        bar3[:, i] = foo .+ i
    end

    @test compute_mse(bar, foo) ≈ [0, 0]
    @test compute_mse(bar2, foo) ≈ [0, 0]
    @test compute_mse(bar3, foo) ≈ [1, 4]
    @test_throws BoundsError compute_mse(bar4, foo)
end

@testset "check_nans" begin
    arr = [1.0, NaN, 3.0]
    arr2 = penalize_nan(arr; penalization = 1.0e5)
    @test arr2 ≈ [1.0, 1.0e5, 3.0]
end

@testset "struct_serializer" begin
    struct MyStruct
        foo::Float64
        bar::Array{Int}
    end

    my_struct = MyStruct(0.1, [2, 3])
    serial_struct = serialize_struct(my_struct)
    @test isa(serial_struct, Dict)
    my_struct_2 = deserialize_struct(serial_struct, MyStruct)
    @test isa(my_struct_2, MyStruct)
    @test my_struct == my_struct_2
end

@testset "netCDF handling" begin
    using CalibrateEDMF.ReferenceModels
    using CalibrateEDMF.TurbulenceConvectionUtils

    pwdir = mktempdir()
    output_root = joinpath(pwdir, "foo/bar/scm")
    case_name_test = "DYCOMS_RF02"
    scm_test_uuid = "12345"
    # Use SCM sim as data
    y_dir_test = joinpath(output_root, "Output.$case_name_test.$scm_test_uuid")
    y_names = ["thetal_mean", "ql_mean", "qt_mean"]
    ti = 0.0
    tf = 10.0

    ref_model = ReferenceModel(y_names, y_dir_test, case_name_test, ti, tf)
    run_reference_SCM(
        ref_model;
        output_root = output_root,
        uuid = scm_test_uuid,
        overwrite = true,
        run_single_timestep = true,
    )
    data_filename = y_nc_file(ref_model)

    # Error handling
    @test_throws ErrorException fetch_interpolate_transform("Michael Scott", data_filename, nothing)
    @test_throws ErrorException fetch_interpolate_transform("tequila_resolved_z_flux", data_filename, nothing)
    # Get height
    zc = get_height(data_filename)
    zf = get_height(data_filename; get_faces = true)
    # TC.jl faces start at z=0
    @test zf[1] ≈ 0
    @test zc[1] > zf[1]
    @test is_face_variable(data_filename, "total_flux_h")
    @test !is_face_variable(data_filename, "u_mean")
end

@testset "namelist utilities" begin
    namelist = Dict(
        "turbulence" => Dict(
            "EDMF_PrognosticTKE" =>
                Dict("sorting_power" => 1.0, "general_stochastic_ent_params" => [1.0, 2.0, 3.0, 4.0]),
        ),
        "microphysics" => Dict(),
        "time_stepping" => Dict("t_max" => 12.0),
    )

    @test namelist_subdict_by_key(namelist, "sorting_power")["sorting_power"] == 1.0
    @test namelist_subdict_by_key(namelist, "t_max") == Dict("t_max" => 12.0)
    @test_throws ArgumentError namelist_subdict_by_key(namelist, "fake_param")
end

@testset "parameter mapping" begin
    namelist = Dict(
        "turbulence" => Dict(
            "EDMF_PrognosticTKE" =>
                Dict("sorting_power" => 1.0, "general_stochastic_ent_params" => [1.0, 2.0, 3.0, 4.0]),
        ),
        "microphysics" => Dict(),
        "time_stepping" => Dict("t_max" => 12.0),
    )
    u_names = ["general_stochastic_ent_params_{1}", "general_stochastic_ent_params_{2}"]
    u = [11.0, 12.0]

    # # do-nothing param map by default
    param_map1 = do_nothing_param_map()
    u_names1, u1 = expand_params(u_names, u, param_map1, namelist)
    @test u_names1 == [u_names..., "general_stochastic_ent_params_{3}", "general_stochastic_ent_params_{4}"]
    @test u1 == [11.0, 12.0, 3.0, 4.0]

    # String and Number mapping
    param_map2 = ParameterMap(;
        mapping = Dict(
            "general_stochastic_ent_params_{3}" => 33.0,
            "general_stochastic_ent_params_{4}" => "general_stochastic_ent_params_{2}",
        ),
    )

    u_names2, u2 = expand_params(u_names, u, param_map2, namelist)
    @test u_names2 == [u_names..., "general_stochastic_ent_params_{3}", "general_stochastic_ent_params_{4}"]
    @test u2 == [11.0, 12.0, 33.0, 12.0]

    # Parameter map to non-existent parameter
    param_map3 = ParameterMap(; mapping = Dict("general_stochastic_ent_params_{4}" => "fake_param"))
    @test_throws AssertionError expand_params(u_names, u, param_map3, namelist)

    # Parameter map to unknown type
    param_map4 = ParameterMap(; mapping = Dict("general_stochastic_ent_params_{4}" => [1, 2]))
    @test_throws ArgumentError expand_params(u_names, u, param_map4, namelist)
end
