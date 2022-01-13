#Ensemble Kalman Inversion: specific structures and function definitions

using Distributions
using LinearAlgebra
using Random
using Statistics

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributionStorage
using EnsembleKalmanProcesses.DataStorage
import EnsembleKalmanProcesses.EnsembleKalmanProcessModule: update_ensemble_prediction!, construct_mean, construct_cov

"""
     split_indices_by_success(g::Array{FT, 2}) where {FT <: Real}

Returns the successful/failed particle indices given a matrix with output vectors stored as columns.
Failures are defined for particles containing at least one NaN output element.
"""
function split_indices_by_success(g::Array{FT, 2}) where {FT <: Real}
    failed_ens = [i for i = 1:size(g, 2) if any(isnan.(g[:, i]))]
    successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
    if length(failed_ens) > length(successful_ens)
        @warn string(
            "More than 50% of runs produced NaNs ($(length(failed_ens))/$(size(g, 2))).",
            "\nIterating... but consider increasing model stability.",
            "\nThis will affect optimization result.",
        )
    end
    return successful_ens, failed_ens
end

"""
     update_successful_ens(
        u::Array{FT, 2},
        g::Array{FT, 2},
        y::Array{FT, 2},
        obs_noise_cov::Matrix{FT},
    ) where {FT <: Real}

Returns the updated parameter vectors for the successful ensemble, ignoring failed particles.
"""
function update_successful_ens(
    u::Array{FT, 2},
    g::Array{FT, 2},
    y::Array{FT, 2},
    obs_noise_cov::Matrix{FT},
) where {FT <: Real}

    cov_ug = cov(u, g, dims = 2, corrected = false) # [N_par × N_obs]
    cov_gg = cov(g, g, dims = 2, corrected = false) # [N_par × N_obs]

    # N_obs × N_obs \ [N_obs × N_ens]
    # --> tmp is [N_obs × N_ens]
    tmp = (cov_gg + obs_noise_cov) \ (y - g)
    return u + (cov_ug * tmp) # [N_par × N_ens]  
end

"""
     update_failed_ens(u_old_fail::Array{FT, 2}, u_succ::Array{FT, 2}, failure_handler::String) where {FT <: Real}

Returns the updated parameter vectors for the failed ensemble, using a given failure handling method.
"""
function update_failed_ens(u_old_fail::Array{FT, 2}, u_succ::Array{FT, 2}, failure_handler::String) where {FT <: Real}
    if failure_handler == "sample_succ_gauss"
        cov_u_new = cov(u_succ, u_succ, dims = 2)
        mean_u_new = mean(u_succ, dims = 2)
        return rand(MvNormal(mean_u_new[:], cov_u_new), size(u_old_fail, 2))
    else
        throw(ArgumentError("Failure handler $failure_handler not recognized."))
    end
end

"""
    update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, <:Inversion}, g::Array{FT,2} cov_threshold::FT=0.01, Δt_new=nothing) where {FT, IT}
Updates the ensemble according to which type of Process we have. Model outputs `g` need to be a `N_obs × N_ens` array (i.e data are columms).
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    g::Array{FT, 2};
    cov_threshold::FT = 0.01,
    Δt_new = nothing,
    deterministic_forward_map = true,
    failure_handler = nothing,
) where {FT, IT}

    # Update follows eqns. (4) and (5) of Schillings and Stuart (2017)

    #catch works when g non-square
    if !(size(g)[2] == ekp.N_ens)
        throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g do not match, try transposing g or check ensemble size"))
    end

    #get successes and failures
    successful_ens, failed_ens = split_indices_by_success(g)

    # u: N_par × N_ens 
    # g: N_obs × N_ens
    u = deepcopy(get_u_final(ekp))
    u_old = deepcopy(u)
    N_obs = size(g, 1)
    cov_init = cov(u, dims = 2)

    if !isnothing(Δt_new)
        push!(ekp.Δt, Δt_new)
    elseif isnothing(Δt_new) && isempty(ekp.Δt)
        push!(ekp.Δt, FT(1))
    else
        push!(ekp.Δt, ekp.Δt[end])
    end

    # Scale noise using Δt
    scaled_obs_noise_cov = ekp.obs_noise_cov / ekp.Δt[end]
    noise = rand(MvNormal(zeros(N_obs), scaled_obs_noise_cov), ekp.N_ens)

    # Add obs_mean (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic
    y = deterministic_forward_map ? (ekp.obs_mean .+ noise) : (ekp.obs_mean .+ zero(noise))

    # No explicit failure handling
    if length(failed_ens) == 0 || isnothing(failure_handler)
        u = update_successful_ens(u, g, y, scaled_obs_noise_cov)

        # Failure handling
    elseif isa(failure_handler, String)
        u[:, successful_ens] = update_successful_ens(
            u[:, successful_ens],
            g[:, successful_ens],
            y[:, successful_ens],
            scaled_obs_noise_cov,
        )
        u[:, failed_ens] = update_failed_ens(
            u[:, failed_ens],       # NB: at t^n
            u[:, successful_ens],   # NB: at t^{n+1}
            failure_handler,
        )
        println("Particle failure(s) detected. Handler used: $failure_handler.")
    end

    # store new parameters (and model outputs)
    push!(ekp.u, DataContainer(u, data_are_columns = true))
    push!(ekp.g, DataContainer(g, data_are_columns = true))

    # Store error
    compute_error!(ekp)

    # Check convergence
    cov_new = cov(get_u_final(ekp), dims = 2)
    cov_ratio = det(cov_new) / det(cov_init)
    if cov_ratio < cov_threshold
        @warn string(
            "New ensemble covariance determinant is less than ",
            cov_threshold,
            " times its previous value.",
            "\nConsider reducing the EK time step.",
        )
    end
end


function update_ensemble_analysis!(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    u_p::Matrix{FT},
    g::Matrix{FT},
    failure_handler::Union{String, Nothing},
) where {FT <: AbstractFloat, IT <: Int}

    obs_mean = uki.obs_mean
    Σ_ν = uki.process.Σ_ν_scale * uki.obs_noise_cov

    ############# Prediction step:

    u_p_mean = construct_mean(uki, u_p)
    uu_p_cov = construct_cov(uki, u_p, u_p_mean)

    ###########  Analysis step

    #get successes and failures
    successful_ens, failed_ens = split_indices_by_success(g)

    if length(failed_ens) == 0 || isnothing(failure_handler)

        g_mean = construct_mean(uki, g)
        gg_cov = construct_cov(uki, g, g_mean) + Σ_ν
        ug_cov = construct_cov(uki, u_p, u_p_mean, g, g_mean)

        # Ignore failed particles and perform analysis using successful particles.
    elseif failure_handler == "sample_succ_gauss"

        g_mean = construct_failsafe_mean(uki, g, successful_ens)
        gg_cov = construct_failsafe_cov(uki, g, g_mean, successful_ens) + Σ_ν
        ug_cov = construct_failsafe_cov(uki, u_p, u_p_mean, g, g_mean, successful_ens)
        println("Particle failure(s) detected. Handler used: $failure_handler.")

    else
        throw(ArgumentError("Failure handler $failure_handler not recognized."))
    end

    tmp = ug_cov / gg_cov

    u_mean = u_p_mean + tmp * (obs_mean - g_mean)
    uu_cov = uu_p_cov - tmp * ug_cov'

    ########### Save results
    push!(uki.process.obs_pred, g_mean) # N_ens x N_data
    push!(uki.process.u_mean, u_mean) # N_ens x N_params
    push!(uki.process.uu_cov, uu_cov) # N_ens x N_data

    push!(uki.g, DataContainer(g, data_are_columns = true))

    compute_error!(uki)
end

"""
Failsafe construct_mean `x_mean` from ensemble `x`.
"""
function construct_failsafe_mean(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::Matrix{FT},
    successful_indices::Union{Vector{IT}, Vector{Any}},
) where {FT <: AbstractFloat, IT <: Int}
    N_x, N_ens = size(x)

    @assert(uki.N_ens == N_ens)

    x_mean = zeros(FT, N_x)

    mean_weights = deepcopy(uki.process.mean_weights)

    # Rescale weights to sum to unity if center did not fail
    if 1 in successful_indices
        mean_weights = mean_weights ./ sum(mean_weights[successful_indices])
        # Equally weighted mean if center particle failed
    else
        mean_weights .= 1 / length(successful_indices)
    end
    for i in 1:N_ens
        if i in successful_indices
            x_mean += mean_weights[i] * x[:, i]
        end
    end

    return x_mean
end

"""
Failsafe construct_cov `xx_cov` from ensemble `x` and mean `x_mean`.
"""
function construct_failsafe_cov(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::Matrix{FT},
    x_mean::Array{FT},
    successful_indices::Union{Vector{IT}, Vector{Any}},
) where {FT <: AbstractFloat, IT <: Int}
    N_ens, N_x = uki.N_ens, size(x_mean, 1)

    cov_weights = deepcopy(uki.process.cov_weights)

    # Rescale non-center sigma weights to sum to original value
    orig_weight_sum = sum(cov_weights[2:end])
    sum_indices = filter(x -> x > 1, successful_indices)
    succ_weight_sum = sum(cov_weights[sum_indices])
    cov_weights[2:end] = cov_weights[2:end] .* (orig_weight_sum / succ_weight_sum)

    xx_cov = zeros(FT, N_x, N_x)

    for i in 1:N_ens
        if i in successful_indices
            xx_cov .+= cov_weights[i] * (x[:, i] - x_mean) * (x[:, i] - x_mean)'
        end
    end

    return xx_cov
end

"""
Failsafe construct_cov `xy_cov` from ensemble x and mean `x_mean`, ensemble `obs_mean` and mean `y_mean`.
"""
function construct_failsafe_cov(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::Matrix{FT},
    x_mean::Array{FT},
    obs_mean::Matrix{FT},
    y_mean::Array{FT},
    successful_indices::Union{Vector{IT}, Vector{Any}},
) where {FT <: AbstractFloat, IT <: Int}
    N_ens, N_x, N_y = uki.N_ens, size(x_mean, 1), size(y_mean, 1)

    cov_weights = deepcopy(uki.process.cov_weights)

    # Rescale non-center sigma weights to sum to original value
    orig_weight_sum = sum(cov_weights[2:end])
    sum_indices = filter(x -> x > 1, successful_indices)
    succ_weight_sum = sum(cov_weights[sum_indices])
    cov_weights[2:end] = cov_weights[2:end] .* (orig_weight_sum / succ_weight_sum)

    xy_cov = zeros(FT, N_x, N_y)

    for i in 1:N_ens
        if i in successful_indices
            xy_cov .+= cov_weights[i] * (x[:, i] - x_mean) * (obs_mean[:, i] - y_mean)'
        end
    end

    return xy_cov
end

function update_ensemble!(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    g_in::Matrix{FT};
    failure_handler::Union{String, Nothing} = nothing,
) where {FT <: AbstractFloat, IT <: Int}
    #catch works when g_in non-square 
    if !(size(g_in)[2] == uki.N_ens)
        throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g_in do not match, try transposing or check ensemble size"))
    end

    u_p_old = get_u_final(uki)

    #perform analysis on the model runs
    update_ensemble_analysis!(uki, u_p_old, g_in, failure_handler)
    #perform new prediction output to model parameters u_p
    u_p = update_ensemble_prediction!(uki.process)

    push!(uki.u, DataContainer(u_p, data_are_columns = true))

    return u_p
end
