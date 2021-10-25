using Test
using Glob
using CalibrateEDMF.Pipeline
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
import TurbulenceConvection
tc_dir = dirname(dirname(pathof(TurbulenceConvection)))
include(joinpath(tc_dir, "integration_tests", "utils", "parameter_set.jl"))

include("config.jl")

@testset "Pipeline" begin
    config = get_config()
    init_calibration(config["process"]["N_ens"], config["process"]["N_iter"], config; mode = "hpc")
    res_dir_list = glob("results_*_scm")
    res_dir = res_dir_list[1]
    # Check for output
    @test isdir("scm_init")
    @test isdir("scm_init/Output.Bomex.000000")
    @test length(res_dir_list) == 1
    @test isdir(res_dir)
    @test isfile(joinpath(res_dir, "prior.jld2"))
    @test isfile(joinpath(res_dir, "ekobj_iter_1.jld2"))
    @test isfile(joinpath(res_dir, "versions_1.txt"))

    version_list = split(open(f -> read(f, String), joinpath(res_dir, "versions_1.txt")), "\n")
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(res_dir, "scm_initializer_$(version_list[i]).jld2"))
    end

    res_dict = init_calibration(config["process"]["N_ens"], config["process"]["N_iter"], config; mode = "pmap")

    @test typeof(res_dict["ekobj"]) == EnsembleKalmanProcess{Float64, Int64, Inversion}
    @test typeof(res_dict["ref_stats"]) == ReferenceStatistics{Float64}
    @test typeof(res_dict["ref_models"]) == Vector{ReferenceModel}
    @test res_dict["n_param"] == 2
end
