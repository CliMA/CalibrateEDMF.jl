"""
    DistributionUtils

Parameter and probability distribution utils.
"""
module DistributionUtils

using Distributions
using JLD2
using EnsembleKalmanProcesses.ParameterDistributions

import ..AbstractTypes: OptVec
import ..HelperFuncs: ParameterMap, do_nothing_param_map, keys_ordered, values_ordered

export construct_priors
export logmean_and_logstd, mean_and_std_from_ln
export flatten_config_dict
export flat_dict_keys_where, identity, above_eps

"""
    flatten_config_dict(param_dict::Dict{String, Vector{T}})

Given a dictionary of parameter names => parameter vectors of arbitrary length,
return a pair of vectors (name, values) with vector components as unique parameters.

Inputs:

 - param_dict :: Dictionary of parameter names to constraints.

Outputs:
 - u_names :: Vector{String} :: vector of parameter names
 - values :: Vector{Vector{T}} :: vector of single-valued vectors encapsulating parameter values.
"""
function flatten_config_dict(param_dict::Dict{String, T}) where {T}

    u_names = Vector{String}()
    values = Vector{T}()
    for param in keys_ordered(param_dict)
        value = param_dict[param]
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

"Boolean specifying whether the input value is significant to machine precision"
above_eps(x::FT) where {FT <: Real} = x > eps(FT)

"Identity function"
identity(x) = x

"""
    flat_dict_keys_where(dict::Dict{String, Vector{T}}, condition::Function = identity) where {T}

Flattens the values of a dictionary with parameter vectors as keys, and returns the indices of
entries in the flattened dictionary satisfying a given condition.

Inputs:
    param_dict :: Dictionary of parameter names to vectors.
    condition :: A condition function operating on each dictionary value.
Outputs:
    Indices of flattened entries satisfying the `condition`.
"""
function flat_dict_keys_where(param_dict::Dict{String, Vector{T}}, condition::Function = identity) where {T}
    # flatten global dict
    if any(1 .< [length(val) for val in collect(values(param_dict))])
        _, dict_values = flatten_config_dict(param_dict)
    else
        dict_values = values_ordered(param_dict)
    end
    dict_values = vcat(dict_values...)

    return findall(x -> condition(x), dict_values)
end

"""
    construct_priors(
        const_dict::Dict{String, T};
        unconstrained_σ::FT = 1.0,
        prior_mean::Union{Dict{String, Vector{Float64}}, Nothing} = nothing,
        outdir_path::String = pwd(),
        to_file::Bool = true,
    ) where {T, FT}

Define a prior Gaussian ParameterDistribution in unconstrained space
from a dictionary of constraints.

This constructor assumes independent priors and the same unconstrained
standard deviation for each parameter. Note that the standard deviation
in unconstrained space is normalized with respect to the constrained
interval width, so it automatically takes into account parameter scales.

The constructor also allows passing a prior mean for each parameter in
constrained space.

Inputs:
 - const_dict :: Dictionary of parameter names to constraints.
 - unconstrained_σ :: Standard deviation of the transformed gaussians (unconstrained space)
    or a dictionary mapping parameter names to standard deviations in unconstrained space.
 - prior_mean :: The mean value of the prior in constrained space. If not given,
    the prior is selected to be 0 in the centered unconstrained space.
 - outdir_path :: Output path.
 - to_file :: Whether to write the serialized prior to a JLD2 file.

Output:
 - The prior ParameterDistribution.
"""
function construct_priors(
    const_dict::Dict{String, T};
    unconstrained_σ::Union{FT, Dict{String, Vector{FT}}} = 1.0,
    prior_mean::Union{Dict{String, Vector{Float64}}, Nothing} = nothing,
    outdir_path::String = pwd(),
    to_file::Bool = true,
) where {T, FT}
    if !isnothing(prior_mean)
        @assert keys(const_dict) == keys(prior_mean)
    end
    if unconstrained_σ isa Dict
        @assert keys(const_dict) == keys(unconstrained_σ)
    end
    # if parameter vectors found => flatten
    if any(1 .< [length(val) for val in collect(values(const_dict))])
        u_names, constraints = flatten_config_dict(const_dict)
        if !isnothing(prior_mean)
            u_names_mean, prior_μ = flatten_config_dict(prior_mean)
            @assert u_names_mean == u_names # ensure order preserved
        else
            prior_μ = nothing
        end
        if unconstrained_σ isa Dict
            u_names_σ, unconstrained_σ = flatten_config_dict(unconstrained_σ)
            @assert u_names_σ == u_names # ensure order preserved
        end
    else
        u_names = keys_ordered(const_dict)
        constraints = values_ordered(const_dict)
        prior_μ = !isnothing(prior_mean) ? values_ordered(prior_mean) : nothing
        unconstrained_σ = unconstrained_σ isa Dict ? values_ordered(unconstrained_σ) : unconstrained_σ
    end
    n_param = length(u_names)
    @assert isnothing(prior_μ) || length(prior_μ) == n_param

    marginal_priors = construct_prior.(u_names, constraints, prior_μ, unconstrained_σ)
    prior = combine_distributions(marginal_priors)
    to_file ? jldsave(joinpath(outdir_path, "prior.jld2"); prior) : nothing
    return prior
end

"""
    construct_prior(
        param_name::String,
        constraint::Vector{CT},
        prior_μ::OptVec{FT},
        unconstrained_σ,
    ) where {CT, FT <: Real}

Define a prior Gaussian ParameterDistribution in unconstrained space
from a constraint, a prior in constrained space, and the standard deviation
in unconstrained space.

The standard deviation in unconstrained space is normalized with respect to
the constrained interval width, so it automatically takes into account parameter scales.

Inputs:
 - param_name :: A parameter name.
 - constraint :: A 1-element vector containing the constraints.
 - prior_μ :: A 1-element vector containing the constrained prior mean.
 - unconstrained_σ :: Standard deviation of the transformed gaussians (unconstrained space).

Output:
 - The prior ParameterDistribution.
"""
function construct_prior(
    param_name::String,
    constraint::Vector{CT},
    prior_μ::OptVec{FT},
    unconstrained_σ::Union{ST, Vector{FT}},
) where {CT, FT <: Real, ST <: Real}
    if unconstrained_σ isa Vector
        unconstrained_σ = unconstrained_σ[1]
    end
    if isnothing(prior_μ)
        distribution = Parameterized(Normal(0.0, unconstrained_σ))
    else
        uncons_μ = constraint[1].constrained_to_unconstrained(prior_μ[1])
        distribution = Parameterized(Normal(uncons_μ, unconstrained_σ))
    end
    return ParameterDistribution(distribution, constraint, param_name)
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
