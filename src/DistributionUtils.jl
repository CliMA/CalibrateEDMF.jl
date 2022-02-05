"""
    DistributionUtils

Parameter and probability distribution utils.
"""
module DistributionUtils

using Distributions
using JLD2
using EnsembleKalmanProcesses.ParameterDistributionStorage

export construct_priors, deserialize_prior
export logmean_and_logstd, mean_and_std_from_ln


"""
    construct_priors(
        params::Dict{String, Vector{Constraint}};
        unconstrained_σ::Float64 = 1.0,
        prior_mean::Union{Vector{Float64}, Nothing} = nothing,
        outdir_path::String = pwd(),
        to_file::Bool = true,
    )

Define a prior Gaussian ParameterDistribution in unconstrained space
from a dictionary of constraints.

This constructor assumes independent priors and the same unconstrained
standard deviation for each parameter. Note that the standard deviation
in unconstrained space is normalized with respect to the constrained
interval width, so it automatically takes into account parameter scales.

The constructor also allows passing a prior mean for each parameter in
constrained space.

Inputs:
 - params :: Dictionary of parameter names to constraints.
 - unconstrained_σ :: Standard deviation of the transformed gaussians (unconstrained space).
 - prior_mean :: The mean value of the prior in constrained space. If not given,
    the prior is selected to be 0 in unconstrained space.
 - outdir_path :: Output path.
 - to_file :: Whether to write the serialized prior to a JLD2 file.

Output:
 - The prior ParameterDistribution.
"""
function construct_priors(
    params::Dict{String, Vector{Constraint}};
    unconstrained_σ::Float64 = 1.0,
    prior_mean::Union{Dict{String, Vector{Float64}}, Nothing} = nothing,
    outdir_path::String = pwd(),
    to_file::Bool = true,
)
    # if parameter vectors found => flatten
    if any(1 .< [length(val) for val in collect(values(params))])
        u_names, constraints = flatten_config_dict(params)
    else
        u_names = collect(keys(params))
        constraints = collect(values(params))
    end

    n_param = length(u_names)

    # All vars are approximated as Gaussian in unconstrained space.
    if isnothing(prior_mean)
        distributions = repeat([Parameterized(Normal(0.0, unconstrained_σ))], n_param)
    else
        if any(1 .< [length(val) for val in collect(values(prior_mean))])
            u_names_mean, prior_mean = flatten_config_dict(prior_mean)
            @assert u_names_mean == u_names
        else
            prior_mean = collect(values(prior_mean))
        end
        @assert length(prior_mean) == n_param
        uncons_prior_mean =
            [c[1].constrained_to_unconstrained(μ_cons[1]) for (μ_cons, c) in zip(prior_mean, constraints)]
        distributions = [Parameterized(Normal(uncons_μ[1], unconstrained_σ)) for uncons_μ in uncons_prior_mean]
    end
    to_file ? jldsave(joinpath(outdir_path, "prior.jld2"); distributions, constraints, u_names) : nothing
    return ParameterDistribution(distributions, constraints, u_names)
end

"""
    flatten_config_dict(param_dict::Dict{String, Vector{T}})

For parameter names that correspond to vectors, assign a unique name to each vector component and treat as independent parameter.
Inputs:
    param_dict :: Dictionary of parameter names to constraints.
Outputs:
    u_names :: Vector{String} :: vector of parameter names
    values :: Vector{Vector{T}} :: vector
"""
function flatten_config_dict(param_dict::Dict{String, Vector{T}}) where {T}

    u_names = Vector{String}()
    values = Vector{Vector{T}}()
    for (param, value) in param_dict
        if length(value) > 1
            for j in 1:length(value)
                push!(u_names, "$(param)_{$j}")
                push!(values, [value[j]])
            end
        else
            push!(u_names, param)
            push!(values, value)
        end
    end
    return (u_names, values)
end

"""
    deserialize_prior(prior_dict::Dict{String, Any})

Generates a prior ParameterDistribution from arguments stored
in a dictionary.
"""
function deserialize_prior(prior_dict::Dict{String, T}) where {T}
    return ParameterDistribution(prior_dict["distributions"], prior_dict["constraints"], prior_dict["u_names"])
end

"""
    logmean_and_logstd(μ, σ)

Returns the lognormal parameters μ and σ from the mean μ and std σ of the 
lognormal distribution.
"""
function logmean_and_logstd(μ, σ)
    σ_log = sqrt(log(1.0 + σ^2 / μ^2))
    μ_log = log(μ / (sqrt(1.0 + σ^2 / μ^2)))
    return μ_log, σ_log
end


"""
    mean_and_std_from_ln(μ, σ)

Returns the mean and variance of the lognormal distribution
from the lognormal parameters μ and σ.
"""
function mean_and_std_from_ln(μ_log, σ_log)
    μ = exp(μ_log + σ_log^2 / 2)
    σ = sqrt((exp(σ_log^2) - 1) * exp(2 * μ_log + σ_log^2))
    return μ, σ
end

end # module
