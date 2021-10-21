using Test
using CalibrateEDMF.TurbulenceConvectionUtils


@testset "TurbulenceConvectionUtils" begin
    @test get_gcm_les_uuid(1, forcing_model = "model1", month = 1, experiment = "experiment1") ==
          "1_model1_01_experiment1"
end
