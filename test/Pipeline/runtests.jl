using Test
using Glob
using Random
using JLD2
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.Pipeline
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
import TurbulenceConvection
tc = dirname(dirname(pathof(TurbulenceConvection)))
include(joinpath(tc, "integration_tests", "utils", "parameter_set.jl"))
src = dirname(pathof(CalibrateEDMF))
include(joinpath(src, "helper_funcs.jl"))
include("config.jl")

# Shared simulations
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

@testset "Pipeline" begin

    ###  Test HPC pipeline
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

    priors = deserialize_prior(load(joinpath(res_dir, "prior.jld2")))
    ekobj = load(ekobj_path(res_dir, 1))["ekp"]
    versions = readlines(joinpath(res_dir, "versions_1.txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(res_dir, "scm_initializer_$(versions[i]).jld2"))
    end

    # Run one simulation and perturb results to emulate ensemble
    scm_args = load(scm_init_path(res_dir, versions[1]))
    model_evaluator = scm_args["model_evaluator"]
    model_evaluator = precondition(model_evaluator, priors, namelist_args = namelist_args)
    sim_dirs, g_scm_orig, g_scm_pca_orig = run_SCM(model_evaluator, namelist_args = namelist_args)
    for version in versions
        g_scm = g_scm_orig .* (1.0 + rand())
        g_scm_pca = g_scm_pca_orig .* (1.0 + rand())
        jldsave(scm_output_path(res_dir, version); sim_dirs, g_scm, g_scm_pca, model_evaluator, version)
        @test isfile(joinpath(res_dir, "scm_output_$version.jld2"))
    end

    ek_update(ekobj, priors, 1, config, versions, res_dir)

    # Test ek_update output
    @test isfile(joinpath(res_dir, "ekobj_iter_2.jld2"))
    @test isfile(joinpath(res_dir, "versions_2.txt"))
    versions = readlines(joinpath(res_dir, "versions_2.txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(res_dir, "scm_initializer_$(versions[i]).jld2"))
    end
    rm(res_dir, recursive = true)
end

@testset "Pipeline_julia" begin
    ### Test pmap pipeline
    res_dict = init_calibration(config; mode = "pmap")

    ekobj = res_dict["ekobj"]
    N_par, N_ens = size(ekobj.u[1])

    res_dir_list = glob("results_*_SCM*", config["output"]["outdir_root"])
    res_dir = res_dir_list[1]
    rm(res_dir, recursive = true)

    @test typeof(ekobj) == EnsembleKalmanProcess{Float64, Int64, Inversion}
    @test typeof(res_dict["ref_stats"]) == ReferenceStatistics{Float64}
    @test typeof(res_dict["ref_models"]) == Vector{ReferenceModel}
    @test N_par == 2
    @test N_ens == 5
end

@testset "Pipeline_minibatch" begin

    config["reference"]["batch_size"] = 1
    config["process"]["algorithm"] = "Unscented"
    init_calibration(config; mode = "hpc")
    res_dir_list = glob("results_*_SCM*", config["output"]["outdir_root"])
    res_dir = res_dir_list[1]
    versions = readlines(joinpath(res_dir, "versions_1.txt"))

    # Run one simulation and perturb results to emulate ensemble
    scm_args = load(scm_init_path(res_dir, versions[1]))
    model_evaluator = scm_args["model_evaluator"]
    sim_dirs, g_scm_orig, g_scm_pca_orig = run_SCM(model_evaluator, namelist_args = namelist_args)
    for version in versions
        g_scm = g_scm_orig .* (1.0 + rand())
        g_scm_pca = g_scm_pca_orig .* (1.0 + rand())
        jldsave(scm_output_path(res_dir, version); sim_dirs, g_scm, g_scm_pca, model_evaluator, version)
        @test isfile(joinpath(res_dir, "scm_output_$version.jld2"))
    end
    @test isfile(joinpath(res_dir, "ref_model_batch.jld2"))
    ref_model_batch = load(joinpath(res_dir, "ref_model_batch.jld2"))["ref_model_batch"]
    @test isa(ref_model_batch.ref_models, Vector{ReferenceModel})
    @test isa(ref_model_batch.eval_order, Vector{Int64})

    priors = deserialize_prior(load(joinpath(res_dir, "prior.jld2")))
    ekobj = load(ekobj_path(res_dir, 1))["ekp"]
    ek_update(ekobj, priors, 1, config, versions, res_dir)

    # Test ek_update output
    @test isfile(joinpath(res_dir, "ekobj_iter_2.jld2"))
    @test isfile(joinpath(res_dir, "versions_2.txt"))
    versions = readlines(joinpath(res_dir, "versions_2.txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(res_dir, "scm_initializer_$(versions[i]).jld2"))
    end
end
