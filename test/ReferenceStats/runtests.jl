using Test
using LinearAlgebra
using Random
using Statistics
using LinearAlgebra
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.HelperFuncs
using CalibrateEDMF.DistributionUtils

@testset "ReferenceStatistics" begin
    # Choose same SCM to speed computation
    data_dir = mktempdir()
    case = "Bomex"
    uuid = "012"
    y_dirs = repeat([joinpath(data_dir, "Output.$case.$uuid")], 2)
    # Reduce resolution and t_max to speed computation as well
    t_max = 4 * 3600.0
    dt_max = 30.0
    dt_min = 20.0
    io_frequency = 120.0
    namelist_args = [
        ("time_stepping", "t_max", t_max),
        ("time_stepping", "dt_max", dt_max),
        ("time_stepping", "dt_min", dt_min),
        ("stats_io", "frequency", io_frequency),
        ("grid", "dz", 150.0),
        ("grid", "nz", 20),
        ("logging", "truncate_stack_trace", true),
    ]

    kwargs_ref_model = Dict(
        :y_names => [["u_mean"], ["v_mean", "lwp_mean"]],
        :y_dir => y_dirs,
        :case_name => repeat(["Bomex"], 2),
        :t_start => repeat([t_max - 2.0 * 3600], 2),
        :t_end => repeat([t_max], 2),
        :namelist_args => repeat([namelist_args], 2),
        :y_type => SCM(),
        :Σ_type => SCM(),
    )
    ref_models = construct_reference_models(kwargs_ref_model)
    run_reference_SCM.(ref_models, output_root = data_dir, uuid = uuid, overwrite = false, run_single_timestep = false)

    filename = get_stats_path(y_dirs[1])
    ref_model = ref_models[1]

    ## Test penalty behavior of ReferenceStatistics.get_profile

    # if simulation lines up with ti, tf => no penalty
    y = ReferenceStats.get_profile(filename, ["u_mean", "v_mean"]; ti = 0.0, tf = t_max)
    @test maximum(y) < 1e4
    # if simulation end within dt margin => no penalty
    y = ReferenceStats.get_profile(filename, ["u_mean", "v_mean"]; ti = 0.0, tf = t_max + io_frequency)
    @test maximum(y) < 1e4
    # if simulation end outside of dt margin => penalty
    y = ReferenceStats.get_profile(filename, ["u_mean", "v_mean"]; ti = 0.0, tf = t_max + io_frequency + dt_max)
    @test all(y .> 1e4)
    # if simulation end after ti  => no penalty
    y = ReferenceStats.get_profile(filename, ["u_mean", "v_mean"]; ti = t_max - 0.01, tf = t_max)
    @test maximum(y) < 1e4
    # if simulation end before ti  => penalty
    y = ReferenceStats.get_profile(filename, ["u_mean", "v_mean"]; ti = t_max + 0.01, tf = t_max)
    @test all(y .> 1e4)
    @test length(y) == 40

    # Test recovery of averaged scalars from timeseries
    y = ReferenceStats.get_profile(filename, ["lwp_mean"]; ti = 0.0, tf = t_max)
    @test length(y) == 1

    # Test profile indices
    y, profile_indices =
        ReferenceStats.get_profile(filename, ["u_mean", "lwp_mean", "v_mean"]; ti = 0.0, tf = t_max, prof_ind = true)
    @test profile_indices == [true, false, true]

    z_obs = get_z_obs(ref_model)
    obs = ["u_mean", "v_mean"]
    Σ_norm, pool_var_1 = ReferenceStats.get_time_covariance(ref_model, obs, z_obs; normalize = true)
    Σ, pool_var_2 = ReferenceStats.get_time_covariance(ref_model, obs, z_obs; normalize = false)
    @assert pool_var_1 ≈ pool_var_2
    @assert !(Σ ≈ Σ_norm)
    @assert tr(Σ_norm) ≈ length(z_obs) * length(obs)
    @assert !(tr(Σ) ≈ length(z_obs) * length(obs))

    # Test only tikhonov vs PCA and tikhonov
    pca_list = [false, true, false]
    norm_list = [false, true, true]
    mode_list = ["absolute", "relative", "relative"]
    dim_list = [false, true, true]
    model_err_list = [nothing, [[0.1], [0.0, 0.0]], [nothing, [0.2, 1.0]]]
    for (perform_PCA, norm, tikhonov_mode, dim_scaling, model_err) in
        zip(pca_list, norm_list, mode_list, dim_list, model_err_list)
        ref_stats = ReferenceStatistics(
            ref_models;
            perform_PCA = perform_PCA,
            normalize = norm,
            variance_loss = 0.1,
            tikhonov_noise = 0.1,
            tikhonov_mode = tikhonov_mode,
            dim_scaling = dim_scaling,
            model_errors = model_err,
        )

        @test pca_length(ref_stats) == size(ref_stats.Γ, 1)
        @test full_length(ref_stats) == size(ref_stats.Γ_full, 1)
        @test (pca_length(ref_stats) < full_length(ref_stats)) == perform_PCA
        for (ci, m) in enumerate(ref_models)
            y_case, _, _ = get_obs(m, norm, z_scm = get_z_obs(m))
            @test full_length(ref_stats, ci) == length(y_case)
            y_case_pca = ref_stats.pca_vec[ci]' * y_case
            @test pca_length(ref_stats, ci) == length(y_case_pca)
            case_full_inds = full_inds(ref_stats, ci)
            case_pca_inds = pca_inds(ref_stats, ci)
            @test ref_stats.y_full[case_full_inds] == y_case
            @test ref_stats.y[case_pca_inds] == y_case_pca
        end
        @test_throws ArgumentError pca_inds(ref_stats, 0)
        @test_throws ArgumentError pca_inds(ref_stats, 3)
        if perform_PCA == false
            # Tikhonov regularization results in variance inflation
            @test tr(ref_stats.Γ) > tr(ref_stats.Γ_full)
            # Since the same configuration is added twice, the full
            # covariance is singular if no model error is added.
            if isnothing(model_err)
                @test det(ref_stats.Γ_full) ≈ 0
            end
            # But not the regularized covariance.
            @test det(ref_stats.Γ) > 0
            for ci in 1:2
                @test pca_length(ref_stats, ci) == full_length(ref_stats, ci)
                @test pca_inds(ref_stats, ci) == full_inds(ref_stats, ci)
            end
        end
    end

    l2_reg = Dict("foo" => [0.1], "bar" => [0.3, 0.0, 0.4])
    reg_indices = flat_dict_keys_where(l2_reg, above_eps)
    @test length(reg_indices) == 3
    # Dict is not ordered
    @test reg_indices == [1, 3, 4]

    @testset "PCA" begin
        dofs = [5, 10, 25, 100, 250]
        ts = [30, 300, 200, 50, 40]
        fs = [0.2, 0.1, 0.05, 0.2, 0.1]
        for (dof, t, f) in zip(dofs, ts, fs)
            time_evol = rand(dof, t)
            μ = vcat(mean(time_evol, dims = 2)...)
            Γ = cov(time_evol, dims = 2)
            λ, eigvecs = eigen(Γ)
            λ_pca, P_pca = pca(Γ, f)
            len_pca = length(λ_pca)
            @test Diagonal(λ_pca) ≈ P_pca' * Γ * P_pca
            @test sum(λ_pca) < tr(Γ)
            @test sum(λ_pca) > (1 - f) * tr(Γ)
            @test sum(λ_pca) - λ[dof - len_pca + 1] < (1 - f) * tr(Γ)

            μ_pca, Γ_pca, P_pca = obs_PCA(μ, Γ, f)
            @test tr(Γ_pca) < tr(Γ)
            @test tr(Γ_pca) ≈ sum(λ_pca)
            @test tr(Γ_pca) > (1 - f) * tr(Γ)
            @test tr(Γ_pca) - λ[dof - len_pca + 1] < (1 - f) * tr(Γ)
        end
    end
end
