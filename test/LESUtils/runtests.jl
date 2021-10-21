using Test
using CalibrateEDMF.LESUtils


@testset "LESUtils" begin
    scm_names = ["total_flux_qt", "total_flux_h", "u_mean"]

    @test get_les_names(scm_names, "GABLS") == ["resolved_z_flux_qt", "resolved_z_flux_theta", "u_translational_mean"]
    @test get_les_names(scm_names, "foo") == ["resolved_z_flux_qt", "resolved_z_flux_thetali", "u_translational_mean"]
end
