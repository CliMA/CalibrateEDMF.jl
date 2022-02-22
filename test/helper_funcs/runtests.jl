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
    arr2 = penalize_nan(arr, penalization = 1.0e5)
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
