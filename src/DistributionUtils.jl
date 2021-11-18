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
        outdir_path::String = pwd(),
        to_file::Bool = true,
    )

Define a prior gaussian ParameterDistribution in unconstrained space
from a dictionary of constraints.

Inputs:
 - params :: Dictionary of parameter names to constraints.
 - unconstrained_σ :: Standard deviation of the transformed gaussians.
 - outdir_path :: Output path.
 - to_file :: Whether to write the serialized prior to a JLD2 file.

Output:
 - The prior ParameterDistribution
"""
function construct_priors(
    params::Dict{String, Vector{Constraint}};
    unconstrained_σ::Float64 = 1.0,
    outdir_path::String = pwd(),
    to_file::Bool = true,
)

    # if parameter vectors found => flatten
    if any(1 .< [length(val) for val in collect(values(params))])
        u_names, constraints = flatten_param_dict(params)
    else
        u_names = collect(keys(params))
        constraints = collect(values(params))
    end

    n_param = length(u_names)
    # All vars are approximately uniform in unconstrained space
    distributions = repeat([Parameterized(Normal(0.0, unconstrained_σ))], n_param)
    to_file ? jldsave(joinpath(outdir_path, "prior.jld2"); distributions, constraints, u_names) : nothing
    return ParameterDistribution(distributions, constraints, u_names)
end

"""
    flatten_param_dict(param_dict::Dict{String, Vector{Constraint}})

For parameter names that correspond to vectors, assign unique name to each vector component.
Inputs:
    param_dict :: Dictionary of parameter names to constraints.
Outputs:
    u_names :: vector of parameter names
    constraints :: vector of constraints
"""
function flatten_param_dict(param_dict::Dict{String, Vector{Constraint}})

    u_names = Vector{String}()
    constraints = Vector{Vector{Constraint}}()
    for (param, value) in param_dict
        if length(value) > 1
            for j in 1:length(value)
                push!(u_names, "$(param)_{$j}")
                push!(constraints, [value[j]])
            end
        else
            push!(u_names, param)
            push!(constraints, [value])
        end
    end
    return (u_names, constraints)
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

log_transform(a::AbstractArray) = log.(a)
exp_transform(a::AbstractArray) = exp.(a)

end # module
