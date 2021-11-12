using Test
using LinearAlgebra
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils

@testset "ReferenceStatistics" begin
    # Choose same SCM to speed computation
    data_dir = mktempdir()
    scm_dirs = repeat([joinpath(data_dir, "Output.Bomex.000000")], 2)
    kwargs_ref_model = Dict(
        :y_names => [["u_mean"], ["v_mean"]],
        :y_dir => scm_dirs,
        :scm_dir => scm_dirs,
        :case_name => repeat(["Bomex"], 2),
        :t_start => repeat([4.0 * 3600], 2),
        :t_end => repeat([6.0 * 3600], 2),
    )
    ref_models = construct_reference_models(kwargs_ref_model)
    run_SCM(ref_models, overwrite = false)

    # Test only tikhonov vs PCA and tikhonov
    pca_list = [false, true]
    norm_list = [false, true]
    mode_list = ["absolute", "relative"]
    dim_list = [false, true]
    for (pca, norm, tikhonov_mode, dim_scaling) in zip(pca_list, norm_list, mode_list, dim_list)
        ref_stats = ReferenceStatistics(
            ref_models,
            pca,
            norm;
            variance_loss = 0.1,
            tikhonov_noise = 0.1,
            tikhonov_mode = tikhonov_mode,
            dim_scaling = dim_scaling,
            y_type = SCM(),
            Σ_type = SCM(),
        )

        @test pca_length(ref_stats) == size(ref_stats.Γ, 1)
        @test full_length(ref_stats) == size(ref_stats.Γ_full, 1)
        @test (pca_length(ref_stats) < full_length(ref_stats)) == pca
        if pca == false
            # Tikhonov regularization results in variance inflation
            @test tr(ref_stats.Γ) > tr(ref_stats.Γ_full)
            # Since the same configuration is added twice,
            # the full covariance is singular,
            @test det(ref_stats.Γ_full) ≈ 0
            # But not the regularized covariance.
            @test det(ref_stats.Γ) > 0
        end
    end

    # Verify that incorrect definitions throw error
    @test_throws AssertionError ReferenceStatistics(
        ref_models,
        false,
        false;
        tikhonov_mode = "relative",
        y_type = SCM(),
        Σ_type = SCM(),
    )
end