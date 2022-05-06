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
using CalibrateEDMF.Pipeline: get_ref_model_kwargs
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
import TurbulenceConvection
cedmf = pkgdir(CalibrateEDMF)
test_dir = joinpath(cedmf, "test", "Pipeline")
using CalibrateEDMF.HelperFuncs
using Random
include(joinpath(test_dir, "config.jl"))

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
run_reference_SCM(ref_model, run_single_timestep = false)

# Test a range of calibration initializations
@testset "init_calibration" begin

    # Different configurations
    prior_means = [
        Dict("entrainment_factor" => [0.15], "detrainment_factor" => [0.4]),
        nothing,
        nothing,
        Dict("entrainment_factor" => [0.1], "detrainment_factor" => [0.2]),
    ]
    batch_sizes = [nothing, 1, 1, nothing]
    augments = [true, true, true, false]
    l2_regs = [1.0, Dict("entrainment_factor" => [0.0], "detrainment_factor" => [0.1]), nothing, nothing]
    norms = [true, false, true, true]
    dim_scalings = [true, false, true, false]
    tikhonov_modes = ["relative", "absolute", "relative", "relative"]
    modes = ["pmap", "hpc", "hpc", "pmap"]
    validations = [config["reference"], config["reference"], config["reference"], nothing]
    algos = ["Sampler", "Unscented", "SparseInversion", "Inversion"]

    for (prior_mean, batch_size, augment, norm, dim_scaling, tikhonov_mode, mode, validation, algo, l2_reg) in
        zip(prior_means, batch_sizes, augments, norms, dim_scalings, tikhonov_modes, modes, validations, algos, l2_regs)
        config["prior"]["prior_mean"] = prior_mean
        config["reference"]["batch_size"] = batch_size
        config["process"]["augmented"] = augment
        config["regularization"]["l2_reg"] = l2_reg
        config["process"]["algorithm"] = algo
        config["regularization"]["normalize"] = norm
        config["regularization"]["dim_scaling"] = dim_scaling
        config["regularization"]["tikhonov_mode"] = tikhonov_mode
        config["validation"] = validation

        outdir_path = init_calibration(config; config_path = joinpath(test_dir, "config.jl"), mode = mode)

        @test isdir(outdir_path)
        @test isfile(joinpath(outdir_path, "prior.jld2"))
        @test isfile(joinpath(outdir_path, "ekobj_iter_1.jld2"))
        @test isfile(joinpath(outdir_path, "versions_1.txt"))
        !isnothing(batch_size) ? (@test isfile(joinpath(outdir_path, "ref_model_batch.jld2"))) : nothing

        versions = readlines(joinpath(outdir_path, "versions_1.txt"))
        for i in 1:config["process"]["N_ens"]
            @test isfile(joinpath(outdir_path, "scm_initializer_$(versions[i]).jld2"))
        end
        rm(outdir_path, recursive = true)
    end
end

@testset "Pipeline with restart" begin

    ###  Test HPC pipeline
    outdir_path = init_calibration(config; mode = "hpc", config_path = joinpath(test_dir, "config.jl"))

    # Check for output
    @test all(isdir.(config["reference"]["scm_parent_dir"]))
    @test isdir(outdir_path)
    @test isfile(joinpath(outdir_path, "prior.jld2"))
    @test isfile(joinpath(outdir_path, "ekobj_iter_1.jld2"))
    @test isfile(joinpath(outdir_path, "versions_1.txt"))

    ekobj = load(ekobj_path(outdir_path, 1))["ekp"]
    versions = readlines(joinpath(outdir_path, "versions_1.txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(outdir_path, "scm_initializer_$(versions[i]).jld2"))
    end

    # Run one simulation and perturb results to emulate ensemble
    scm_args = load(scm_init_path(outdir_path, versions[1]))
    batch_indices = scm_args["batch_indices"]
    priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))
    model_evaluator = scm_args["model_evaluator"]
    model_evaluator = precondition(model_evaluator, priors, namelist_args = namelist_args)
    sim_dirs, g_scm_orig, g_scm_pca_orig = run_SCM(model_evaluator, namelist_args = namelist_args)
    for version in versions
        g_scm = g_scm_orig .* (1.0 + rand())
        g_scm_pca = g_scm_pca_orig .* (1.0 + rand())
        jldsave(
            scm_output_path(outdir_path, version);
            sim_dirs,
            g_scm,
            g_scm_pca,
            model_evaluator,
            version,
            batch_indices,
        )
        @test isfile(joinpath(outdir_path, "scm_output_$version.jld2"))
    end

    ek_update(ekobj, priors, 1, config, versions, outdir_path)

    # Test ek_update output
    @test isfile(joinpath(outdir_path, "ekobj_iter_2.jld2"))
    @test isfile(joinpath(outdir_path, "versions_2.txt"))
    versions = readlines(joinpath(outdir_path, "versions_2.txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(outdir_path, "scm_initializer_$(versions[i]).jld2"))
    end

    restart_calibration(ekobj, priors, 2, config, outdir_path, mode = "hpc")
    @test isfile(joinpath(outdir_path, "ekobj_iter_3.jld2"))
    @test isfile(joinpath(outdir_path, "versions_3.txt"))
    versions = readlines(joinpath(outdir_path, "versions_3.txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(outdir_path, "scm_initializer_$(versions[i]).jld2"))
    end

    rm(outdir_path, recursive = true)
end

@testset "Pipeline_with_validation" begin

    config["reference"]["batch_size"] = 1
    config["reference"]["n_obs"] = [10]
    config["process"]["algorithm"] = "Unscented"
    config["validation"] = config["reference"]
    config["validation"]["batch_size"] = 1
    config["process"]["augmented"] = true
    config["regularization"]["l2_reg"] = 0.5
    init_calibration(config; mode = "hpc", config_path = joinpath(test_dir, "config.jl"))
    outdir_path_list = glob("results_*_SCM*", config["output"]["outdir_root"])
    outdir_path = outdir_path_list[1]
    versions = readlines(joinpath(outdir_path, "versions_1.txt"))

    # Run one simulation and perturb results to emulate ensemble
    scm_args = load(scm_init_path(outdir_path, versions[1]))
    batch_indices = scm_args["batch_indices"]
    model_evaluator = scm_args["model_evaluator"]
    sim_dirs, g_scm_orig, g_scm_pca_orig = run_SCM(model_evaluator, namelist_args = namelist_args)
    for version in versions
        g_scm = g_scm_orig .* (1.0 + rand())
        g_scm_pca = g_scm_pca_orig .* (1.0 + rand())
        jldsave(
            scm_output_path(outdir_path, version);
            sim_dirs,
            g_scm,
            g_scm_pca,
            model_evaluator,
            version,
            batch_indices,
        )
        jldsave(
            scm_val_output_path(outdir_path, version);
            sim_dirs,
            g_scm,
            g_scm_pca,
            model_evaluator,
            version,
            batch_indices,
        )
        @test isfile(joinpath(outdir_path, "scm_output_$version.jld2"))
        @test isfile(joinpath(outdir_path, "scm_val_output_$version.jld2"))
    end
    @test isfile(joinpath(outdir_path, "ref_model_batch.jld2"))
    ref_model_batch = load(joinpath(outdir_path, "ref_model_batch.jld2"))["ref_model_batch"]
    @test isa(ref_model_batch.ref_models, Vector{ReferenceModel})
    @test isa(ref_model_batch.eval_order, Vector{Int64})
    @test isfile(joinpath(outdir_path, "val_ref_model_batch.jld2"))
    val_ref_model_batch = load(joinpath(outdir_path, "ref_model_batch.jld2"))["ref_model_batch"]
    @test isa(val_ref_model_batch.ref_models, Vector{ReferenceModel})
    @test isa(val_ref_model_batch.eval_order, Vector{Int64})

    priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))
    ekobj = load(ekobj_path(outdir_path, 1))["ekp"]
    ek_update(ekobj, priors, 1, config, versions, outdir_path)

    # Test ek_update output
    @test isfile(joinpath(outdir_path, "ekobj_iter_2.jld2"))
    @test isfile(joinpath(outdir_path, "versions_2.txt"))
    versions = readlines(joinpath(outdir_path, "versions_2.txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(outdir_path, "scm_initializer_$(versions[i]).jld2"))
    end

    restart_calibration(ekobj, priors, 2, config, outdir_path, mode = "hpc")
    @test isfile(joinpath(outdir_path, "ekobj_iter_3.jld2"))
    @test isfile(joinpath(outdir_path, "versions_3.txt"))
    versions = readlines(joinpath(outdir_path, "versions_3.txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(outdir_path, "scm_initializer_$(versions[i]).jld2"))
    end
end
