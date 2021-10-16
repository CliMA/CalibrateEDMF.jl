using Test
using Distributions
using CalibrateEDMF.DistributionUtils
using EnsembleKalmanProcesses.ParameterDistributionStorage

@testset "Priors" begin
    tmpdir = mktempdir()

    params = Dict("foo" => [bounded(0.1, 1.0)], "bar" => [bounded(0.2, 2.0)])
    priors1 = construct_priors(params, outdir_path = tmpdir, to_file = false, unconstrained_σ = 1.0)

    @test isempty(readdir(tmpdir))
    # Construction from Dict is ordered from keys
    @test priors1.names == ["bar", "foo"]

    priors2 = construct_priors(params, outdir_path = tmpdir)

    @test !isempty(readdir(tmpdir))
    @test readdir(tmpdir)[1] == "prior.jld2"

    # Construction from lists is ordered
    prior_dict = Dict(
        "u_names" => ["bar", "foo"],
        "constraints" => [[bounded(0.2, 2.0)], [bounded(0.1, 1.0)]],
        "distributions" => repeat([Parameterized(Normal(0.0, 1.0))], 2),
    )
    priors3 = deserialize_prior(prior_dict)

    @test priors1.names == priors3.names
    @test priors1.constraints == priors3.constraints
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