using Test
using Distributions
using CalibrateEDMF.DistributionUtils
using EnsembleKalmanProcesses.ParameterDistributionStorage

@testset "Priors" begin
    tmpdir = mktempdir()

    params = Dict("foo" => [bounded(0.1, 1.0)], "bar" => [bounded(0.2, 2.0)])
    priors1 = construct_priors(params, outdir_path = tmpdir, to_file = false, unconstrained_σ = 1.0)

    @test isempty(readdir(tmpdir))
    # Construction from Dict does not retain ordering
    @test priors1.names == ["bar", "foo"]

    # Construction from lists (deserializer) is ordered
    prior_dict = Dict(
        "u_names" => ["foo", "bar"],
        "constraints" => [[bounded(0.1, 1.0)], [bounded(0.2, 2.0)]],
        "distributions" => repeat([Parameterized(Normal(0.0, 1.0))], 2),
    )
    priors2 = deserialize_prior(prior_dict)

    @test priors2.names == ["foo", "bar"]
    @test priors2.names != priors1.names
    @test priors2.constraints != priors1.constraints

    μ_correct = [0.4, 0.5]
    μ_incorrect1 = [0.0, 2.5] # Out of bounds
    μ_incorrect2 = [0.1] # Incorrect shape
    priors3 = construct_priors(params, outdir_path = tmpdir, prior_mean = μ_correct)

    @test !isempty(readdir(tmpdir))
    @test readdir(tmpdir)[1] == "prior.jld2"
    @test_throws AssertionError construct_priors(params, outdir_path = tmpdir, prior_mean = μ_incorrect2)
    @test_throws DomainError construct_priors(params, outdir_path = tmpdir, prior_mean = μ_incorrect1)
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
