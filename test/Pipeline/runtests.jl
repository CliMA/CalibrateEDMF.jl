using Test
using Glob
using CalibrateEDMF.Pipeline
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
import TurbulenceConvection
tc = dirname(dirname(pathof(TurbulenceConvection)))
include(joinpath(tc, "integration_tests", "utils", "parameter_set.jl"))

include("config.jl")

@testset "Pipeline" begin
    config = get_config()
    # Generate reference data
    ref_config = config["reference"]
    ref_model = ReferenceModel(
        ref_config["y_names"][1],
        ref_config["y_dir"][1],
        ref_config["y_dir"][1],
        ref_config["case_name"][1],
        ref_config["t_start"][1],
        ref_config["t_end"][1],
    )
    namelist_args = config["scm"]["namelist_args"]
    # Generate "true" data
    run_reference_SCM(ref_model, run_single_timestep = false, namelist_args = namelist_args)
    # Initialize calibration setup
    init_calibration(config; mode = "hpc")
    res_dir_list = glob("results_*_SCM*", config["output"]["outdir_root"])
    res_dir = res_dir_list[1]
    # Check for output
    @test all(isdir.(config["reference"]["scm_parent_dir"]))
    @test length(res_dir_list) == 1
    @test isdir(res_dir)
    @test isfile(joinpath(res_dir, "prior.jld2"))
    @test isfile(joinpath(res_dir, "ekobj_iter_1.jld2"))
    @test isfile(joinpath(res_dir, "versions_1.txt"))

    version_list = split(open(f -> read(f, String), joinpath(res_dir, "versions_1.txt")), "\n")
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(res_dir, "scm_initializer_$(version_list[i]).jld2"))
    end

    # Re-use previous simulation
    res_dict = init_calibration(config; mode = "pmap")

    ekobj = res_dict["ekobj"]
    N_par, N_ens = size(ekobj.u[1])

    @test typeof(ekobj) == EnsembleKalmanProcess{Float64, Int64, Inversion}
    @test typeof(res_dict["ref_stats"]) == ReferenceStatistics{Float64}
    @test typeof(res_dict["ref_models"]) == Vector{ReferenceModel}
    @test N_par == 2
    @test N_ens == 5
end
