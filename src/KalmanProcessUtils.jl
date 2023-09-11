"""
    KalmanProcessUtils

Utils for the construction and handling of Kalman Process structs.
"""
module KalmanProcessUtils

export generate_ekp,
    generate_tekp, get_sparse_indices, get_regularized_indices, get_Δt, modify_field, update_ek_timestepper, PiecewiseConstantDecay, PiecewiseConstantGrowth

using LinearAlgebra
using Statistics
using JLD2
using DocStringExtensions
using EnsembleKalmanProcesses
import EnsembleKalmanProcesses: Process, Unscented, Inversion, Sampler, SparseInversion
import EnsembleKalmanProcesses: SampleSuccGauss, IgnoreFailures
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
import EnsembleKalmanProcesses.Localizers: LocalizationMethod, NoLocalization, Delta, RBF, BernoulliDropout

using ..ReferenceModels
using ..ReferenceStats
using ..ModelTypes
using ..LESUtils
using ..HelperFuncs
using ..DistributionUtils

abstract type LearningRateScheduler end

"""
    PiecewiseConstantDecay{FT <: Real, IT <: Int} <: LearningRateScheduler

Piecewise constant decay learning rate scheduler.

Halves the time step periodically with period `τ`.

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct PiecewiseConstantDecay{FT <: Real, IT <: Int} <: LearningRateScheduler
    "Initial learning rate"
    Δt_init::FT
    "Halving time"
    τ::IT
end

"""
    PiecewiseConstantGrowth{FT <: Real, IT <: Int} <: LearningRateScheduler

Piecewise constant growth learning rate scheduler.

Doubles the time step periodically with period `τ`.

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct PiecewiseConstantGrowth{FT <: Real, IT <: Int} <: LearningRateScheduler
    "Initial learning rate"
    Δt_init::FT
    "Doubling time"
    τ::IT
end

"Retrieve learning rate `Δt` from a LearningRateScheduler"
get_Δt(Δt::FT, iteration::IT) where {FT <: Real, IT <: Int} = Δt
get_Δt(lrs::PiecewiseConstantDecay, iteration::IT) where {IT <: Int} = lrs.Δt_init * 2^(-floor(iteration / lrs.τ))
get_Δt(lrs::PiecewiseConstantGrowth, iteration::IT) where {IT <: Int} = lrs.Δt_init * 2^(floor(iteration / lrs.τ))

"""
    generate_ekp(
        ref_stats::ReferenceStatistics,
        process::Process,
        u::Union{Matrix{T}, T} = nothing;
        failure_handler::String = "ignore_failures",
        localizer::LocalizationMethod = NoLocalization(),
        outdir_path::String = pwd(),
        to_file::Bool = true,
    ) where {T}

Generates, and possible writes to file, an EnsembleKalmanProcess
from a parameter ensemble and reference statistics.

Inputs:
 - ref_stats :: ReferenceStatistics defining the inverse problem.
 - process :: Type of EnsembleKalmanProcess used to evolve the ensemble.
 - u :: An ensemble of parameter vectors, used if !isa(process, Unscented).
 - failure_handler :: String describing what failure handler to use.
 - localizer :: Covariance localization method.
 - outdir_path :: Output path.
 - to_file :: Whether to write the serialized prior to a JLD2 file.

Output:
 - The generated EnsembleKalmanProcess.
"""
function generate_ekp(
    ref_stats::ReferenceStatistics,
    process::Process,
    u::Union{Matrix{T}, T} = nothing;
    failure_handler::String = "ignore_failures",
    localizer::LocalizationMethod = NoLocalization(),
    scheduler = DefaultScheduler(), 
    outdir_path::String = pwd(),
    to_file::Bool = true,
) where {T}

    @assert isa(process, Unscented) || !isnothing(u) "Incorrect EKP constructor."
    @assert failure_handler in ["ignore_failures", "high_loss", "sample_succ_gauss"]
    if failure_handler == "sample_succ_gauss"
        fh = SampleSuccGauss()
    else
        fh = IgnoreFailures()
    end

    kwargs = Dict(:failure_handler_method => fh, :localization_method => localizer, :scheduler => scheduler)
    ekp =
        isnothing(u) ? EnsembleKalmanProcess(ref_stats.y, ref_stats.Γ, process; kwargs...) :
        EnsembleKalmanProcess(u, ref_stats.y, ref_stats.Γ, process; kwargs...)


    if to_file
        jldsave(ekobj_path(outdir_path, 1); ekp)
    end
    return ekp
end

"""
    generate_tekp(
        ref_stats::ReferenceStatistics,
        priors::ParameterDistribution,
        process::Process,
        u::Union{Matrix{T}, T} = nothing;
        l2_reg::Union{Dict{String, Vector{R}}, R} = nothing,
        failure_handler::String = "ignore_failures",
        localizer::LocalizationMethod = NoLocalization(),
        outdir_path::String = pwd(),
        to_file::Bool = true,
    ) where {T, R}

Generates, and possible writes to file, a Tikhonov EnsembleKalmanProcess
from a parameter ensemble and reference statistics.

Tikhonov regularization is implemented through output state augmentation
with the input parameter vector. The input L2 regularization hyperparameter
should be interpreted as the inverse of the variance of our prior belief in
the magnitude of the parameters.

Inputs:
 - ref_stats :: ReferenceStatistics defining the inverse problem.
 - priors :: Parameter priors used for L2 (i.e., Tikhonov) regularization
 - process :: Type of EnsembleKalmanProcess used to evolve the ensemble.
 - u :: An ensemble of parameter vectors, used if !isa(process, Unscented).
 - l2_reg :: L2 regularization hyperparameter driving parameter values toward prior.
        May be a float (isotropic regularization) or a dictionary of regularizations
        per parameter.
 - failure_handler :: String describing what failure handler to use.
 - localizer :: Covariance localization method.
 - outdir_path :: Output path.
 - to_file :: Whether to write the serialized prior to a JLD2 file.

Output:
 - The generated augmented EnsembleKalmanProcess.
"""
function generate_tekp(
    ref_stats::ReferenceStatistics,
    priors::ParameterDistribution,
    process::Process,
    u::Union{Matrix{T}, T} = nothing;
    l2_reg::Union{Dict{String, Vector{R}}, R} = nothing,
    failure_handler::String = "ignore_failures",
    localizer::LocalizationMethod = NoLocalization(),
    scheduler = DefaultScheduler(),
    outdir_path::String = pwd(),
    to_file::Bool = true,
) where {T, R}

    @assert isa(process, Unscented) || !isnothing(u) "Incorrect TEKP constructor."
    @assert failure_handler in ["ignore_failures", "high_loss", "sample_succ_gauss"]
    if failure_handler == "sample_succ_gauss"
        fh = SampleSuccGauss()
    else
        fh = IgnoreFailures()
    end

    μ = vcat(mean(priors)...)
    if isa(l2_reg, Dict)
        # flatten l2_reg dict
        if any(1 .< [length(val) for val in collect(values(l2_reg))])
            _, l2_reg_values = flatten_config_dict(l2_reg)
        else
            l2_reg_values = values_ordered(l2_reg)
        end
        l2_reg_values = vcat(l2_reg_values...)

        # dict must be complete to preserve ordering
        @assert length(μ) == length(l2_reg_values) "Dictionary of regularizations l2_reg must include all parameters."

        # Augment exclusively with nonzero weights
        aug_indices = get_regularized_indices(l2_reg)
        l2_reg_values = l2_reg_values[aug_indices]
        μ = μ[aug_indices]
        Γ_θ = inv(Diagonal(l2_reg_values))

    elseif !isnothing(l2_reg)
        @assert l2_reg > eps(R) "If system is augmented, provide nonzero l2_reg."
        Γ_θ = Diagonal(repeat([inv(l2_reg)], length(μ)))
    else
        Γ_θ = cov(priors)
    end

    # Augment system with regularization towards prior mean
    y_aug = vcat([ref_stats.y, μ]...)
    Γ_aug_list = [ref_stats.Γ, Array(Γ_θ)]
    Γ_aug = cat(Γ_aug_list..., dims = (1, 2))

    kwargs = Dict(:failure_handler_method => fh, :localization_method => localizer, :scheduler => scheduler)
    ekp =
        isnothing(u) ? EnsembleKalmanProcess(y_aug, Γ_aug, process; kwargs...) :
        EnsembleKalmanProcess(u, y_aug, Γ_aug, process; kwargs...)
    if to_file
        jldsave(ekobj_path(outdir_path, 1); ekp)
    end
    return ekp
end

"Returns the sparse parameter indices given the sparsity configuration and the number of parameters."
function get_sparse_indices(sparsity_config, p)
    if isa(sparsity_config, Dict{String, Vector{Bool}})
        @assert sum(map(length, collect(values(sparsity_config)))) == p
        return flat_dict_keys_where(sparsity_config)
    elseif isa(sparsity_config, Bool) && sparsity_config
        return Colon()
    else
        throw(ArgumentError("Sparsity config entry not recognized, pass a Bool or Dict{String, Vector{Bool}}."))
    end
end

"Returns the indices of parameters to be regularized, given the l2 regularization configuration dictionary."
get_regularized_indices(l2_config::Dict) = flat_dict_keys_where(l2_config, above_eps)

function modify_field(ekp::EnsembleKalmanProcess, field_name::Symbol, new_value)
    fields = fieldnames(EnsembleKalmanProcess)
    values = [field_name == f ? new_value : getfield(ekp, f) for f in fields]
    return EnsembleKalmanProcess(values...)
end

function modify_field(scheduler::T, field_name::Symbol, new_value) where T
    fields = fieldnames(T)
    values = [field_name == f ? new_value : getfield(scheduler, f) for f in fields]
    return T(values...)
end

"""Return new ekp object with updated scheduler object."""
function update_ek_timestepper(ekp, iteration)
    ekp_cp = deepcopy(ekp)
    
    if !isempty(ekp_cp.Δt)
        FT = eltype(ekp_cp.Δt)
        # create empty Δt so that calculate_timestep! updates inv_sqrt_noise
        ekp_cp = modify_field(ekp_cp, :Δt, Vector{FT}())
    end
    scheduler_cp = deepcopy(ekp_cp.scheduler)
    g_ = ekp_cp.obs_noise_cov
    # compute inv_sqrt_noise from obs_mean in current batch
    calculate_timestep!(ekp_cp, g_, nothing, scheduler_cp)
    # update iteration number
    scheduler_updated = modify_field(scheduler_cp, :iteration, [iteration])
    ekp = modify_field(ekp, :scheduler, scheduler_updated)
    return ekp
end



end # module
