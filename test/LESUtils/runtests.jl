using Test
using NCDatasets
using CalibrateEDMF.LESUtils

import CalibrateEDMF.LESUtils: get_LES_library, get_shallow_LES_library

@testset "LESUtils" begin
    scm_names = ["total_flux_qt", "total_flux_h", "u_mean", "foo"]
    tmpdir = mktempdir()
    filename = joinpath(tmpdir, "stats", "test.nc")
    mkdir(dirname(filename))
    NCDataset(filename, "c") do ds

        defDim(ds, "t", 10)
        defDim(ds, "z", 20)

        defGroup(ds, "profiles")
        defVar(ds.group["profiles"], "qt_flux_z", Float32, ("t", "z"))
        defVar(ds.group["profiles"], "resolved_z_flux_thetali", Float32, ("t", "z"))
        defVar(ds.group["profiles"], "u_translational_mean", Float32, ("t", "z"))
        defGroup(ds, "timeseries")
        defVar(ds.group["timeseries"], "foo", Float32, ("t",))
    end

    @test find_alias(("u_mean", "u_translational_mean"), filename) == "u_translational_mean"
    @test find_alias(("foo",), filename) == "foo"
    @test_throws ErrorException find_alias(("vorticity",), filename)
    @test get_les_names(scm_names, filename) == ["qt_flux_z", "resolved_z_flux_thetali", "u_translational_mean", "foo"]

    @testset "cfSite_getter" begin
        @test isa(get_cfsite_les_dir(2), String)
        @test isa(get_cfsite_les_dir(14), String)
        @test isa(get_cfsite_les_dir(15), String)
        @test_throws AssertionError get_cfsite_les_dir(15; month = 4, experiment = "amip4K")
        @test_throws AssertionError get_cfsite_les_dir(15; month = 1, experiment = "amip4K")
        @test isa(get_cfsite_les_dir(94; month = 7, experiment = "amip4K"), String)
        @test_throws AssertionError get_cfsite_les_dir(94; month = 10, experiment = "amip4K")
        @test_throws AssertionError get_cfsite_les_dir(75)
        @test_throws AssertionError get_cfsite_les_dir(2, forcing_model = "foo")
    end

    @testset "artifact_getter" begin
        art_dir = get_path_to_artifact()
        @test isdir(art_dir)
        @test_throws KeyError get_path_to_artifact("foo")
    end

    @testset "LES_library" begin
        shallow_lib = get_shallow_LES_library()
        lib = get_LES_library()
        for month in ["1", "4", "7", "10"]
            @test length(shallow_lib["HadGEM2-A"]["10"]["cfsite_numbers"]) <
                  length(lib["HadGEM2-A"]["10"]["cfsite_numbers"])
        end
    end
end
