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
    scm_dirs = repeat([joinpath(data_dir, "Output.Bomex.000000")], 2)
    # Reduce resolution and t_max to speed computation as well
    t_max = 4 * 3600.0
    dt_max = 30.0
    dt_min = 20.0
    io_frequency = 120.0
    namelist_args = [
        ("time_stepping", "t_max", t_max),
        ("time_stepping", "dt_max", dt_max),
        ("time_stepping", "dt_min", dt_min),
        ("grid", "dz", 150.0),
        ("grid", "nz", 20),
        ("stats_io", "frequency", io_frequency),
    ]
    kwargs_ref_model = Dict(
        :y_names => [["u_mean"], ["v_mean"]],
        :y_dir => scm_dirs,
        :scm_dir => scm_dirs,
        :case_name => repeat(["Bomex"], 2),
        :t_start => repeat([t_max - 2.0 * 3600], 2),
        :t_end => repeat([t_max], 2),
    )
    ref_models = construct_reference_models(kwargs_ref_model)
    run_reference_SCM.(ref_models, overwrite = false, run_single_timestep = false, namelist_args = namelist_args)

    # Test penalty behavior of ReferenceStatistics.get_profile
    filename = get_stats_path(scm_dirs[1])

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

    # Test only tikhonov vs PCA and tikhonov
    pca_list = [false, true]
    norm_list = [false, true]
    mode_list = ["absolute", "relative"]
    dim_list = [false, true]
    for (perform_PCA, norm, tikhonov_mode, dim_scaling) in zip(pca_list, norm_list, mode_list, dim_list)
        ref_stats = ReferenceStatistics(
            ref_models;
            perform_PCA = perform_PCA,
            normalize = norm,
            variance_loss = 0.1,
            tikhonov_noise = 0.1,
            tikhonov_mode = tikhonov_mode,
            dim_scaling = dim_scaling,
            y_type = SCM(),
            Σ_type = SCM(),
        )

        @test pca_length(ref_stats) == size(ref_stats.Γ, 1)
        @test full_length(ref_stats) == size(ref_stats.Γ_full, 1)
        @test (pca_length(ref_stats) < full_length(ref_stats)) == perform_PCA
        for (ci, m) in enumerate(ref_models)
            y_case, _, _ = get_obs(m, SCM(), SCM(), norm, z_scm = get_z_obs(m))
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
            # Since the same configuration is added twice,
            # the full covariance is singular,
            @test det(ref_stats.Γ_full) ≈ 0
            # But not the regularized covariance.
            @test det(ref_stats.Γ) > 0
            for ci in 1:2
                @test pca_length(ref_stats, ci) == full_length(ref_stats, ci)
                @test pca_inds(ref_stats, ci) == full_inds(ref_stats, ci)
            end
        end
    end

    # Verify that incorrect definitions throw error
    @test_throws AssertionError ReferenceStatistics(
        ref_models;
        perform_PCA = false,
        normalize = false,
        tikhonov_mode = "relative",
        y_type = SCM(),
        Σ_type = SCM(),
    )

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
