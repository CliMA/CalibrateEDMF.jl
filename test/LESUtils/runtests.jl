using Test
using NCDatasets
using CalibrateEDMF.LESUtils


@testset "LESUtils" begin
    scm_names = ["total_flux_qt", "total_flux_h", "u_mean", "foo"]
    tmpdir = mktempdir()
    mkdir(joinpath(tmpdir, "stats"))
    NCDataset(joinpath(tmpdir, "stats", "test.nc"), "c") do ds

        defDim(ds, "t", 10)
        defDim(ds, "z", 20)

        defGroup(ds, "profiles")
        defVar(ds.group["profiles"], "qt_flux_z", Float32, ("t", "z"))
        defVar(ds.group["profiles"], "resolved_z_flux_thetali", Float32, ("t", "z"))
        defVar(ds.group["profiles"], "u_translational_mean", Float32, ("t", "z"))
        defGroup(ds, "timeseries")
        defVar(ds.group["timeseries"], "foo", Float32, ("t",))
    end

    @test find_alias(("u_mean", "u_translational_mean"), tmpdir) == "u_translational_mean"
    @test find_alias(("foo",), tmpdir) == "foo"
    @test_throws ErrorException find_alias(("vorticity",), tmpdir)
    @test get_les_names(scm_names, tmpdir) == ["qt_flux_z", "resolved_z_flux_thetali", "u_translational_mean", "foo"]
end

@testset "cfSite_getter" begin
    @test isa(get_cfsite_les_dir(2), String)
    @test_throws AssertionError get_cfsite_les_dir(30)
    @test_throws AssertionError get_cfsite_les_dir(2, forcing_model = "foo")
end
