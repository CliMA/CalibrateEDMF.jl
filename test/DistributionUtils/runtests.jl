using Test
using Distributions
using CalibrateEDMF.DistributionUtils
using EnsembleKalmanProcesses.ParameterDistributions

@testset "Priors" begin
    tmpdir = mktempdir()

    params = Dict("foo" => [bounded(0.1, 1.0)], "bar" => [bounded(0.2, 2.0)])
    priors1 = construct_priors(params, outdir_path = tmpdir, to_file = false, unconstrained_σ = 1.0)

    @test isempty(readdir(tmpdir))
    # Construction from Dict does not retain ordering
    @test priors1.name == ["bar", "foo"]

    # Construction from lists (deserializer) is ordered
    prior_dict = Dict(
        "u_names" => ["foo", "bar"],
        "constraints" => [[bounded(0.1, 1.0)], [bounded(0.2, 2.0)]],
        "distributions" => repeat([Parameterized(Normal(0.0, 1.0))], 2),
    )
    priors2 = deserialize_prior(prior_dict)

    @test priors2.name == ["foo", "bar"]
    @test priors2.name != priors1.name
    @test priors2.constraint != priors1.constraint

    μ_correct = Dict("foo" => [0.4], "bar" => [0.5])
    μ_incorrect1 = Dict("foo" => [0.0], "bar" => [2.5]) # Out of bounds
    μ_incorrect2 = Dict("foo" => [0.1]) # Incorrect shape
    priors3 = construct_priors(params, outdir_path = tmpdir, prior_mean = μ_correct)

    @test !isempty(readdir(tmpdir))
    @test readdir(tmpdir)[1] == "prior.jld2"
    @test_throws AssertionError construct_priors(params, outdir_path = tmpdir, prior_mean = μ_incorrect2)
    @test_throws DomainError construct_priors(params, outdir_path = tmpdir, prior_mean = μ_incorrect1)
end

@testset "Prior Vectors" begin
    tmpdir = mktempdir()
    unc_σ = 0.5

    # vector and float -> correct
    params = Dict("foo" => [bounded(0.1, 1.0)], "bar_vect" => [no_constraint(), bounded(-3.0, 3.0), no_constraint()])
    prior_μ = Dict("foo" => [0.4], "bar_vect" => [1.0, 2.0, 3.0])
    priors = construct_priors(params, outdir_path = tmpdir, unconstrained_σ = unc_σ, prior_mean = prior_μ)

    @test priors.name == ["bar_vect_{1}", "bar_vect_{2}", "bar_vect_{3}", "foo"]
    @test priors.constraint == [no_constraint(), bounded(-3.0, 3.0), no_constraint(), bounded(0.1, 1.0)]
    @test priors.distribution[1].distribution.μ == 1.0
    @test priors.distribution[2].distribution.μ ==
          params["bar_vect"][2].constrained_to_unconstrained(prior_μ["bar_vect"][2])
    @test priors.distribution[4].distribution.μ == params["foo"][1].constrained_to_unconstrained(prior_μ["foo"][1])

    # length mismatch [length(bar_vect mu) != length(bar_vect constraint)]
    params = Dict("foo" => [bounded(0.1, 1.0)], "bar_vect" => [repeat([no_constraint()], 3)...])
    prior_μ = Dict("foo" => [0.4], "bar_vect" => ones(2))
    @test_throws AssertionError construct_priors(
        params,
        outdir_path = tmpdir,
        unconstrained_σ = unc_σ,
        prior_mean = prior_μ,
    )

    # unspecified prior mean should yield μ=0
    params = Dict("foo" => [bounded(0.1, 1.0)], "bar_vect" => [repeat([no_constraint()], 3)...])
    priors = construct_priors(params, outdir_path = tmpdir, unconstrained_σ = unc_σ)
    @test [priors.distribution[i].distribution.μ for i in 1:length(priors.distribution)] == zeros(length(priors.distribution))

end

@testset "Transformations" begin
    μ_log = 1.0
    σ_log = 0.5
    μ, σ = mean_and_std_from_ln(μ_log, σ_log)
    μ_log2, σ_log2 = logmean_and_logstd(μ, σ)

    # Test recovery
    @test σ_log ≈ σ_log2
    @test μ_log ≈ μ_log2

end

@testset "flatten_config_dict" begin
    foo_constraint = [bounded(0.1, 1.0)]
    bar_constraint = [bounded(0.2, 2.0)]
    vect_constraint = [repeat([bounded(-1.0, 1.0)], 2)..., bounded(-0.5, 0.5)]

    params = Dict("foo" => foo_constraint, "bar" => bar_constraint, "vect" => vect_constraint)
    flattened_names, flattened_values = DistributionUtils.flatten_config_dict(params)
    @test flattened_names == ["bar", "vect_{1}", "vect_{2}", "vect_{3}", "foo"]
    @test flattened_values[1] == bar_constraint
    @test flattened_values[2][1] == vect_constraint[1]
    @test flattened_values[4][1] == vect_constraint[3]
    @test flattened_values[5] == foo_constraint
end
