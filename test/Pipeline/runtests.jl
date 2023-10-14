using Test
using Glob
using Random
using JLD2
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.KalmanProcessUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.Pipeline
using CalibrateEDMF.HelperFuncs
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
import TurbulenceConvection
cedmf = pkgdir(CalibrateEDMF)
test_dir = joinpath(cedmf, "test", "Pipeline")
using CalibrateEDMF.HelperFuncs
using Random
include(joinpath(test_dir, "config.jl"))
include(joinpath(test_dir, "config_batching.jl"))

# Shared simulations
config = get_config()
config_batch = get_config_batch()

output_roots = []
for i in 1:2
    # Generate reference data
    ref_config = config_batch["reference"]
    y_dir = ref_config["y_dir"][i]
    ref_model = ReferenceModel(
        ref_config["y_names"][i],
        y_dir,
        ref_config["case_name"][i],
        ref_config["t_start"][i],
        ref_config["t_end"][i];
    )
    # Generate "true" data
    output_root = dirname(y_dir)
    uuid = last(split(y_dir, "."))
    if !ispath(y_dir)
        run_reference_SCM(ref_model; output_root = output_root, uuid = uuid, run_single_timestep = false)
    end
    push!(output_roots, output_root)
end

"""Perturbs results from single SCM simulation to emulate ensemble."""
function generate_SCM_runs(scm_args, outdir_path, versions, priors; precondition_ek::Bool = true)
    batch_indices = scm_args["batch_indices"]
    model_evaluator = scm_args["model_evaluator"]
    if precondition_ek
        model_evaluator = precondition(model_evaluator, priors)
    end
    sim_dirs, g_scm_orig, g_scm_pca_orig = run_SCM(model_evaluator)

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
    end
end

function update_cov_store!(ekp, obs_noise_cov_all, inv_sqrt_noise_all)
    if typeof(ekp.scheduler) <: DataMisfitController
        inv_sqrt_Γ = inv(sqrt(posdef_correct(ekp.obs_noise_cov)))
        @test ekp.scheduler.inv_sqrt_noise[end] == inv_sqrt_Γ
        push!(obs_noise_cov_all, ekp.obs_noise_cov)
        push!(inv_sqrt_noise_all, ekp.scheduler.inv_sqrt_noise[end])
    end
end

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

        temp_file = tempname(pwd())
        test_id = basename(temp_file)
        outdir_path =
            init_calibration(config; config_path = joinpath(test_dir, "config.jl"), mode = mode, job_id = test_id)

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
    temp_file = tempname(pwd())
    test_id = basename(temp_file)
    config = get_config()
    N_iter = config["process"]["N_iter"]
    outdir_path =
        init_calibration(config; mode = "hpc", config_path = joinpath(test_dir, "config.jl"), job_id = test_id)
    priors = load(joinpath(outdir_path, "prior.jld2"))["prior"]
    # Check for output
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
    generate_SCM_runs(scm_args, outdir_path, versions, priors; precondition_ek = true)
    for version in versions
        @test isfile(joinpath(outdir_path, "scm_output_$version.jld2"))
    end

    @info "Running EK updates for $N_iter iterations"
    for iteration in 1:N_iter
        @info "   iter = $iteration"
        # precondition for first iteration
        precondition_ek = iteration == 1
        versions = readlines(joinpath(outdir_path, "versions_$(iteration).txt"))
        scm_args = load(scm_init_path(outdir_path, versions[1]))
        # Run one simulation and perturb results to emulate ensemble
        generate_SCM_runs(scm_args, outdir_path, versions, priors; precondition_ek = precondition_ek)
        ekp = load(ekobj_path(outdir_path, iteration))["ekp"]

        # perform update
        ek_update(ekp, priors, iteration, config, versions, outdir_path)

        if iteration < N_iter
            ekp_next = load(ekobj_path(outdir_path, iteration + 1))["ekp"]

            @test length(ekp_next.Δt) == iteration
            @test isfile(joinpath(outdir_path, "ekobj_iter_$(iteration + 1).jld2"))
            @test isfile(joinpath(outdir_path, "versions_$(iteration + 1).txt"))
            for i in 1:config["process"]["N_ens"]
                @test isfile(joinpath(outdir_path, "scm_initializer_$(versions[i]).jld2"))
            end
        end
    end

    # mimick restart with 2 additional iterations
    config["process"]["N_iter"] = N_iter + 2

    # infer last iteration from files, as in `driver/global_parallel/restart.jl`
    ekobjs = glob("ekobj_iter_*.jld2", outdir_path)
    iters = @. parse(Int64, getfield(match(r"(?<=ekobj_iter_)(\d+)", basename(ekobjs)), :match))
    last_iteration = maximum(iters)
    @test last_iteration == N_iter

    ekobj_prev = load(ekobj_path(outdir_path, last_iteration))["ekp"]

    # test restart
    restart_calibration(ekobj_prev, priors, last_iteration, config, outdir_path; mode = "hpc", job_id = test_id)
    @test isfile(joinpath(outdir_path, "ekobj_iter_$(N_iter + 1).jld2"))
    @test isfile(joinpath(outdir_path, "versions_$(N_iter + 1).txt"))
    versions = readlines(joinpath(outdir_path, "versions_$(N_iter + 1).txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(outdir_path, "scm_initializer_$(versions[i]).jld2"))
    end

    ekobj_restart = load(ekobj_path(outdir_path, N_iter + 1))["ekp"]
    @test length(ekobj_restart.Δt) == N_iter
    @test length(ekobj_restart.u) == N_iter + 1

    rm(outdir_path, recursive = true)
end

@testset "Pipeline with Different Learning Rate Schedulers" begin

    ###  Test HPC pipeline with EKP-native learning rate schedulers
    temp_file = tempname(pwd())
    test_id = basename(temp_file)
    schedulers =
        [nothing, DefaultScheduler(0.75), DataMisfitController(on_terminate = "continue"), EKSStableScheduler()]

    for scheduler in schedulers
        config = get_config()
        N_iter = config["process"]["N_iter"]
        config["process"]["scheduler"] = scheduler
        outdir_path =
            init_calibration(config; mode = "hpc", config_path = joinpath(test_dir, "config.jl"), job_id = test_id)
        ekobj = load(ekobj_path(outdir_path, 1))["ekp"]
        versions = readlines(joinpath(outdir_path, "versions_1.txt"))

        # check that correct learning rate scheduler is used
        initial_obs_noise_cov = deepcopy(ekobj.obs_noise_cov)
        if isnothing(scheduler)
            @test ekobj.scheduler == DefaultScheduler(1.0)
        else
            @test ekobj.scheduler == scheduler
        end
        @test isempty(ekobj.Δt) # test ekp Δt isn't modified before update

        priors = load(joinpath(outdir_path, "prior.jld2"))["prior"]

        # Calibration process
        @info "Running EK updates for $N_iter iterations"
        for iteration in 1:N_iter
            @info "   iter = $iteration"
            # precondition for first iteration
            precondition_ek = iteration == 1
            versions = readlines(joinpath(outdir_path, "versions_$(iteration).txt"))
            scm_args = load(scm_init_path(outdir_path, versions[1]))
            # Run one simulation and perturb results to emulate ensemble
            generate_SCM_runs(scm_args, outdir_path, versions, priors; precondition_ek = precondition_ek)
            ekp = load(ekobj_path(outdir_path, iteration))["ekp"]

            # perform update
            ek_update(ekp, priors, iteration, config, versions, outdir_path)

            if iteration < N_iter
                ekp_next = load(ekobj_path(outdir_path, iteration + 1))["ekp"]
            else
                ekp_next = load(ekobj_path(outdir_path, iteration))["ekp"]
            end

            # check that scheduler is updated appropriately
            if typeof(ekp_next.scheduler) <: DataMisfitController
                # timestep and scheduler are updated in ek_update, so check that previous iteration is correct
                @test ekp_next.scheduler.iteration[end] == iteration
                if iteration < N_iter
                    @test length(ekp_next.scheduler.inv_sqrt_noise) == iteration + 1
                end
            end
            @test length(ekp_next.Δt) == iteration

        end

        # mimick restart with 2 additional iterations
        config["process"]["N_iter"] = N_iter + 2

        # infer last iteration from files, as in `driver/global_parallel/restart.jl`
        ekobjs = glob("ekobj_iter_*.jld2", outdir_path)
        iters = @. parse(Int64, getfield(match(r"(?<=ekobj_iter_)(\d+)", basename(ekobjs)), :match))
        last_iteration = maximum(iters)
        @test last_iteration == N_iter

        ekobj_prev = load(ekobj_path(outdir_path, last_iteration))["ekp"]

        restart_calibration(ekobj_prev, priors, last_iteration, config, outdir_path; mode = "hpc", job_id = test_id)
        ekobj_restart = load(ekobj_path(outdir_path, N_iter + 1))["ekp"]
        # ensure scheduler is unchanged after restart
        @test ekobj_restart.scheduler == ekobj_prev.scheduler

        # make an update after restart (iteration = original N_iter + 1)
        versions = readlines(joinpath(outdir_path, "versions_$(N_iter+1).txt"))
        scm_args = load(scm_init_path(outdir_path, versions[1]))
        generate_SCM_runs(scm_args, outdir_path, versions, priors; precondition_ek = false)
        ek_update(ekobj_restart, priors, N_iter + 1, config, versions, outdir_path)
        ekobj_final = load(ekobj_path(outdir_path, N_iter + 2))["ekp"]

        if typeof(ekobj_final.scheduler) <: DataMisfitController
            # check for expected update behavior after restart
            @test ekobj_final.scheduler.iteration[end] == N_iter + 1
            @test length(ekobj_final.scheduler.inv_sqrt_noise) == N_iter + 2

            # if no batching, obs_noise_cov & inv_sqrt_noise are constant across iterations
            @test length(unique(ekobj_final.scheduler.inv_sqrt_noise)) == 1
        end
        @test length(ekobj_final.Δt) == N_iter + 1

        # check that timesteps are constant for default scheduler
        if typeof(ekobj_final.scheduler) <: DefaultScheduler
            if isnothing(scheduler)
                @test all(ekobj_final.Δt .== 1.0)
            else
                @test all(ekobj_final.Δt .== 0.75)
            end
        end

        # ensure obs_noise_cov remains unchanged
        @test initial_obs_noise_cov == ekobj_final.obs_noise_cov

        rm(outdir_path, recursive = true)
    end

end

@testset "Pipeline_with_validation" begin

    N_iter = 2
    config = get_config()
    config["process"]["N_iter"] = N_iter
    config["prior"]["param_map"] =
        ParameterMap(Dict("sorting_power" => "entrainment_factor", "updraft_mixing_frac" => 0.5))
    config["reference"]["batch_size"] = 1
    config["reference"]["n_obs"] = [10]
    config["process"]["algorithm"] = "Unscented"
    config["validation"] = config["reference"]
    config["validation"]["batch_size"] = 1
    config["process"]["augmented"] = true
    config["regularization"]["l2_reg"] = 0.5
    temp_file = tempname(pwd())
    test_id = basename(temp_file)
    init_calibration(config; mode = "hpc", config_path = joinpath(test_dir, "config.jl"), job_id = test_id)
    outdir_path_list = glob("results_*_SCM*", config["output"]["outdir_root"])
    outdir_path = outdir_path_list[1]
    versions = readlines(joinpath(outdir_path, "versions_1.txt"))
    init_ekp = load(ekobj_path(outdir_path, 1))["ekp"]
    init_obs_noise = deepcopy(init_ekp.obs_noise_cov)

    # Run one simulation and perturb results to emulate ensemble
    scm_args = load(scm_init_path(outdir_path, versions[1]))
    batch_indices = scm_args["batch_indices"]
    model_evaluator = scm_args["model_evaluator"]
    sim_dirs, g_scm_orig, g_scm_pca_orig = run_SCM(model_evaluator)
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

    priors = load(joinpath(outdir_path, "prior.jld2"))["prior"]
    ekobj = load(ekobj_path(outdir_path, 1))["ekp"]
    ek_update(ekobj, priors, 1, config, versions, outdir_path)

    # Test ek_update output
    @test isfile(joinpath(outdir_path, "ekobj_iter_2.jld2"))
    ekobj = load(joinpath(outdir_path, "ekobj_iter_2.jld2"))["ekp"]
    @test ekobj.obs_noise_cov == init_obs_noise
    # The following two tests verifies that a non-trivial parameter map (defined abouve) 
    # has no impact on the dimensionality of the EKP.
    @test size(ekobj.u[1].stored_data) == (2, 5)
    @test length(ekobj.process.u_mean[1]) == 2

    @test isfile(joinpath(outdir_path, "ekobj_iter_2.jld2"))
    @test isfile(joinpath(outdir_path, "versions_2.txt"))
    versions = readlines(joinpath(outdir_path, "versions_2.txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(outdir_path, "scm_initializer_$(versions[i]).jld2"))
    end

    # mimick restart with 1 additional iteration
    config["process"]["N_iter"] = N_iter + 1

    # infer last iteration from files, as in `driver/global_parallel/restart.jl`
    ekobjs = glob("ekobj_iter_*.jld2", outdir_path)
    iters = @. parse(Int64, getfield(match(r"(?<=ekobj_iter_)(\d+)", basename(ekobjs)), :match))
    last_iteration = maximum(iters)
    @test last_iteration == N_iter

    ekobj_prev = load(ekobj_path(outdir_path, last_iteration))["ekp"]

    restart_calibration(ekobj_prev, priors, last_iteration, config, outdir_path; mode = "hpc", job_id = test_id)
    @test isfile(joinpath(outdir_path, "ekobj_iter_$(N_iter + 1).jld2"))
    @test isfile(joinpath(outdir_path, "versions_$(N_iter + 1).txt"))
    versions = readlines(joinpath(outdir_path, "versions_$(N_iter + 1).txt"))
    for i in 1:config["process"]["N_ens"]
        @test isfile(joinpath(outdir_path, "scm_initializer_$(versions[i]).jld2"))
    end
end



@testset "Pipeline with variable timestep, batching & restart" begin

    ###  Test HPC pipeline
    temp_file = tempname(pwd())
    test_id = basename(temp_file)
    config = get_config_batch()
    N_iter = config["process"]["N_iter"]
    outdir_path =
        init_calibration(config; mode = "hpc", config_path = joinpath(test_dir, "config.jl"), job_id = test_id)
    priors = load(joinpath(outdir_path, "prior.jld2"))["prior"]
    ekobj_init = load(ekobj_path(outdir_path, 1))["ekp"]
    versions = readlines(joinpath(outdir_path, "versions_1.txt"))

    # Run one simulation and perturb results to emulate ensemble
    scm_args = load(scm_init_path(outdir_path, versions[1]))
    generate_SCM_runs(scm_args, outdir_path, versions, priors; precondition_ek = true)

    obs_noise_cov_all = []
    inv_sqrt_noise_all = []
    if typeof(ekobj_init.scheduler) <: DataMisfitController
        push!(obs_noise_cov_all, ekobj_init.obs_noise_cov)
        push!(inv_sqrt_noise_all, inv(sqrt(posdef_correct(ekobj_init.obs_noise_cov))))
    end

    @info "Running EK updates for $N_iter iterations"
    for iteration in 1:N_iter
        @info "   iter = $iteration"
        # precondition for first iteration
        precondition_ek = iteration == 1
        versions = readlines(joinpath(outdir_path, "versions_$(iteration).txt"))
        scm_args = load(scm_init_path(outdir_path, versions[1]))
        # Run one simulation and perturb results to emulate ensemble
        generate_SCM_runs(scm_args, outdir_path, versions, priors; precondition_ek = precondition_ek)
        ekp = load(ekobj_path(outdir_path, iteration))["ekp"]

        # ensure inv_sqrt_noise corresponds to current obs_noise_cov
        # note: inv_sqrt_noise empty before first iteration
        if iteration > 1
            update_cov_store!(ekp, obs_noise_cov_all, inv_sqrt_noise_all)
        end

        # perform update
        ek_update(ekp, priors, iteration, config, versions, outdir_path)

        if iteration < N_iter
            ekp_next = load(ekobj_path(outdir_path, iteration + 1))["ekp"]
            @test length(ekp_next.Δt) == iteration
        end
    end

    # mimick restart with 2 additional iterations
    config["process"]["N_iter"] = N_iter + 2

    # infer last iteration from files, as in `driver/global_parallel/restart.jl`
    ekobjs = glob("ekobj_iter_*.jld2", outdir_path)
    iters = @. parse(Int64, getfield(match(r"(?<=ekobj_iter_)(\d+)", basename(ekobjs)), :match))
    last_iteration = maximum(iters)
    @test last_iteration == N_iter

    ekobj_prev = load(ekobj_path(outdir_path, last_iteration))["ekp"]

    if typeof(ekobj_prev.scheduler) <: DataMisfitController
        @test length(ekobj_prev.scheduler.inv_sqrt_noise) == N_iter
        @test length(ekobj_prev.Δt) == N_iter
    end
    # test restart
    restart_calibration(ekobj_prev, priors, last_iteration, config, outdir_path; mode = "hpc", job_id = test_id)

    ekobj_restart = load(ekobj_path(outdir_path, N_iter + 1))["ekp"]
    @test length(ekobj_restart.Δt) == N_iter

    if N_iter > 1
        update_cov_store!(ekobj_restart, obs_noise_cov_all, inv_sqrt_noise_all)
    end

    # make an update after restart (iteration = original N_iter + 1)
    versions = readlines(joinpath(outdir_path, "versions_$(N_iter+1).txt"))
    scm_args = load(scm_init_path(outdir_path, versions[1]))
    generate_SCM_runs(scm_args, outdir_path, versions, priors; precondition_ek = false)
    ek_update(ekobj_restart, priors, N_iter + 1, config, versions, outdir_path)
    ekobj_final = load(ekobj_path(outdir_path, N_iter + 2))["ekp"]

    update_cov_store!(ekobj_final, obs_noise_cov_all, inv_sqrt_noise_all)

    # check that inv_sqrt_noise is correct across all iterations
    for i in 1:length(obs_noise_cov_all)
        inv_sqrt_Γ = inv(sqrt(posdef_correct(obs_noise_cov_all[i])))
        @test inv_sqrt_noise_all[i] == inv_sqrt_Γ
        @test ekobj_final.scheduler.inv_sqrt_noise[i] == inv_sqrt_Γ
    end

    rm(outdir_path, recursive = true)
end


rm(output_roots[1], recursive = true)
