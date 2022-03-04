module ReferenceStats

using SparseArrays
using Statistics
using Interpolations
using LinearAlgebra
using JLD2
# EKP modules
using EnsembleKalmanProcesses
import EnsembleKalmanProcesses: Process, Unscented, Inversion, Sampler
import EnsembleKalmanProcesses: SampleSuccGauss, IgnoreFailures
using EnsembleKalmanProcesses.ParameterDistributions
using ..ReferenceModels
using ..ModelTypes
using ..LESUtils
using ..HelperFuncs
using ..DistributionUtils

export ReferenceStatistics
export pca_length, full_length
export get_obs, get_profile, obs_PCA, pca
export generate_ekp, generate_tekp
export regularized_param_indices


"""
    struct ReferenceStatistics{FT <: Real}

A structure containing statistics from the reference model used to
define a well-posed inverse problem.
"""
Base.@kwdef struct ReferenceStatistics{FT <: Real}
    "Reference data, length: nSim * n_vars * n_zLevels(possibly reduced by PCA)"
    y::Vector{FT}
    "Data covariance matrix, dims: (y,y) (possibly reduced by PCA)"
    Γ::Matrix{FT}
    "Vector (length: nSim) of normalizing factors (length: n_vars)"
    norm_vec::Vector{Vector{FT}}
    "Vector (length: nSim) of PCA projection matrices with leading eigenvectors as columns"
    pca_vec::Vector{Union{Matrix{FT}, UniformScaling}}
    "Full reference data vector, length: nSim * n_vars * n_zLevels"
    y_full::Vector{FT}
    "Full covariance matrix, dims: (y,y)"
    Γ_full::SparseMatrixCSC{FT, Int64}

    """
        ReferenceStatistics(
            RM::Vector{ReferenceModel};
            perform_PCA::Bool = true,
            normalize::Bool = true,
            variance_loss::FT = 0.1,
            tikhonov_noise::FT = 0.0,
            tikhonov_mode::String = "absolute",
            dim_scaling::Bool = false,
            y_type::ModelType = LES(),
            Σ_type::ModelType = LES(),
            Δt::FT = 6 * 3600.0,
        ) where {FT <: Real}

    Constructs the ReferenceStatistics defining the inverse problem.

    Inputs:
     - RM               :: Vector of `ReferenceModel`s.
     - perform_PCA      :: Boolean specifying whether to perform PCA.
     - normalize        :: Boolean specifying whether to normalize the data.
     - variance_loss    :: Fraction of variance loss when performing PCA.
     - tikhonov_noise   :: Tikhonov regularization factor for covariance matrices.
     - tikhonov_mode    :: If "relative", tikhonov_noise is scaled by the maximum
        eigenvalue in the covariance matrix considered, having the interpretation of
        the inverse of the desired condition number. This value is enforced to be
        larger than the sqrt of the machine precision for stability.
     - dim_scaling      :: Whether to scale covariance blocks by their size.
     - y_type           :: Type of reference mean data. Either LES() or SCM().
     - Σ_type           :: Type of reference covariance data. Either LES() or SCM().
     - Δt               :: [LES last time - SCM start time (LES timeframe)] for
       LES_driven_SCM cases.
    Outputs:
     - A ReferenceStatistics struct.
    """
    function ReferenceStatistics(
        RM::Vector{ReferenceModel};
        perform_PCA::Bool = true,
        normalize::Bool = true,
        variance_loss::FT = 0.1,
        tikhonov_noise::FT = 0.0,
        tikhonov_mode::String = "absolute",
        dim_scaling::Bool = false,
        y_type::ModelType = LES(),
        Σ_type::ModelType = LES(),
        Δt::FT = 6 * 3600.0,
    ) where {FT <: Real}
        # Init arrays
        y = FT[]
        Γ_vec = Matrix{FT}[]
        y_full = FT[]
        Γ_full_vec = Matrix{FT}[]
        pca_vec = []
        norm_vec = Vector[]

        for m in RM
            model = m.case_name == "LES_driven_SCM" ? time_shift_reference_model(m, Δt) : m
            # Get (interpolated and pool-normalized) observations, get pool variance vector
            y_, y_var_, pool_var = get_obs(model, y_type, Σ_type, normalize, z_scm = get_z_obs(m))
            push!(norm_vec, pool_var)
            if perform_PCA
                y_pca, y_var_pca, P_pca = obs_PCA(y_, y_var_, variance_loss)
                append!(y, y_pca)
                push!(Γ_vec, y_var_pca)
                push!(pca_vec, P_pca)
            else
                append!(y, y_)
                push!(Γ_vec, y_var_)
                push!(pca_vec, 1.0I)
            end
            # Save full dimensionality (normalized) output for error computation
            append!(y_full, y_)
            push!(Γ_full_vec, y_var_)
        end

        # Construct global observational covariance matrix, original space
        Γ_full = sparse(cat(Γ_full_vec..., dims = (1, 2)))

        # Scale by number of dimensions (averaging loss per dimension)
        Γ_vec = dim_scaling ? length(Γ_vec) .* map(x -> size(x, 1) * x, Γ_vec) : Γ_vec
        # Construct global observational covariance matrix, PCA
        Γ = cat(Γ_vec..., dims = (1, 2))
        # Condition global covariance matrix, PCA
        if tikhonov_mode == "relative"
            @assert perform_PCA "Relative Tikhonov mode only available after PCA change of basis."
            tikhonov_noise = max(tikhonov_noise, 10 * sqrt(eps(FT)))
            Γ = Γ + tikhonov_noise * maximum(diag(Γ)) * I
        else
            Γ = Γ + tikhonov_noise * I
        end

        @assert isposdef(Γ) "Covariance matrix Γ is ill-conditioned, consider regularization."

        return new{FT}(y, Γ, norm_vec, pca_vec, y_full, Γ_full)
    end

    ReferenceStatistics(y::Vector{FT}, args...) where {FT <: Real} = new{FT}(y, args...)
end

pca_length(RS::ReferenceStatistics) = length(RS.y)
full_length(RS::ReferenceStatistics) = length(RS.y_full)

"""
    get_obs(
        m::ReferenceModel,
        y_names::Vector{String},
        Σ_names::Vector{String},
        normalize::Bool;
        z_scm::Union{Vector{FT}, Nothing} = nothing,
    )

Get observations for variables y_names, interpolated to
z_scm (if given), and possibly normalized with respect to the pooled variance.

Inputs:
 - obs_type     :: Either :les or :scm
 - m            :: Reference model
 - z_scm        :: If given, interpolate LES observations to given levels.
Outputs:
 - y            :: Mean of observations, possibly interpolated to z_scm levels.
 - Σ            :: Observational covariance matrix, possibly pool-normalized.
 - pool_var     :: Vector of vertically averaged time-variance, one entry for each variable
"""
function get_obs(
    m::ReferenceModel,
    y_names::Vector{String},
    Σ_names::Vector{String},
    normalize::Bool;
    z_scm::Union{Vector{FT}, Nothing} = nothing,
) where {FT <: Real}
    # time covariance
    Σ, pool_var = get_time_covariance(m, Σ_names, z_scm = z_scm)
    # normalization
    norm_vec = normalize ? pool_var : ones(size(pool_var))
    # Get true observables
    y = get_profile(m, y_nc_file(m), y_names, z_scm = z_scm)
    # normalize
    y = normalize_profile(y, num_vars(m), norm_vec)
    return y, Σ, norm_vec
end

function get_obs(
    m::ReferenceModel,
    y_type::Union{LES, SCM},
    Σ_type::Union{LES, SCM},
    normalize::Bool;
    z_scm::Union{Vector{FT}, Nothing},
) where {FT <: Real}
    y_names = isa(y_type, LES) ? get_les_names(m, y_nc_file(m)) : m.y_names
    Σ_names = isa(Σ_type, LES) ? get_les_names(m, Σ_nc_file(m)) : m.y_names
    get_obs(m, y_names, Σ_names, normalize, z_scm = z_scm)
end

"""
    obs_PCA(y_mean, y_var, allowed_var_loss = 1.0e-1)

Perform dimensionality reduction using principal component analysis on
the variance y_var. Only eigenvectors with eigenvalues that contribute
to the leading 1-allowed_var_loss variance are retained.
Inputs:
 - y_mean           :: Mean of the observations.
 - y_var            :: Variance of the observations.
 - allowed_var_loss :: Maximum variance loss allowed.
Outputs:
 - y_pca            :: Projection of y_mean onto principal subspace spanned by eigenvectors.
 - y_var_pca        :: Projection of y_var on principal subspace.
 - P_pca            :: Projection matrix onto principal subspace, with leading eigenvectors as columns.
"""
function obs_PCA(y_mean::Vector{FT}, y_var::Matrix{FT}, allowed_var_loss::FT = 1.0e-1) where {FT <: Real}
    λ_pca, P_pca = pca(y_var, allowed_var_loss)
    # Project mean
    y_pca = P_pca' * y_mean
    y_var_pca = Diagonal(λ_pca)
    return y_pca, y_var_pca, P_pca
end

"""
    pca(covmat::AbstractMatrix{FT}, allowed_var_loss::FT) where {FT <: Real}

Perform dimensionality reduction using principal component analysis on
the variance covmat.

Inputs:
 - covmat           :: Variance of the observations.
 - allowed_var_loss :: Maximum variance loss allowed.
Outputs:
 - λ_pca            :: Principal eigenvalues, ordered in increasing value order.
 - P_pca            :: Projection matrix onto principal subspace, with leading eigenvectors as columns.
"""
function pca(covmat::AbstractMatrix{FT}, allowed_var_loss::FT) where {FT <: Real}
    eigvals, eigvecs = eigen(covmat)
    # Get index of leading eigenvalues, eigvals are ordered from low to high in julia
    # This expression recovers 1 extra eigenvalue compared to threshold
    leading_eigs = findall(<(1.0 - allowed_var_loss), -cumsum(eigvals) / sum(eigvals) .+ 1)
    P_pca = eigvecs[:, leading_eigs]
    λ_pca = eigvals[leading_eigs]
    return λ_pca, P_pca
end

"""
    get_profile(
        filename::String,
        var_names::Vector{String};
        ti::Real = 0.0,
        tf = nothing,
        z_scm::Union{Vector{FT}, Nothing} = nothing,
    )

Get profiles for variables var_names, interpolated to
z_scm (if given), and concatenated into a single output vector.

Inputs:
 - filename  :: nc filename
 - var_names   :: Names of variables to be retrieved.
 - z_scm :: If given, interpolate LES observations to given levels.
Outputs:
 - y :: Output vector used in the inverse problem, which concatenates the
   requested profiles.
"""
function get_profile(
    filename::String,
    var_names::Vector{String};
    ti::Real = 0.0,
    tf::Union{Real, Nothing} = nothing,
    z_scm::Union{Vector{T}, T} = nothing,
) where {T}

    t = nc_fetch(filename, "t")
    dt = length(t) > 1 ? mean(diff(t)) : 0.0
    y = zeros(0)

    # Check that times are contained in simulation output
    Δt_start, ti_index = findmin(broadcast(abs, t .- ti))
    # If simulation does not contain values for ti or tf, return high value (penalization)
    if t[end] < ti
        @warn string(
            "Note: t_end < ti, which means that simulation stopped before reaching the requested t_start.",
            "Requested t_start = $ti s. However, the last time available is $(t[end]) s.",
            "Defaulting to penalized profiles...",
        )
        for i in 1:length(var_names)
            var_ = isnothing(z_scm) ? get_height(filename) : z_scm
            append!(y, 1.0e5 * ones(length(var_[:])))
        end
        return y
    end
    if !isnothing(tf)
        Δt_end, tf_index = findmin(broadcast(abs, t .- tf))
        if t[end] < tf - dt
            @warn string(
                "Note: t_end < tf - dt, which means that simulation stopped before reaching the requested t_end.",
                "Requested t_end = $tf s. However, the last time available is $(t[end]) s.",
                "Defaulting to penalized profiles...",
            )
            for i in 1:length(var_names)
                var_ = isnothing(z_scm) ? get_height(filename) : z_scm
                append!(y, 1.0e5 * ones(length(var_[:])))
            end
            return y
        end
    end

    # Return time average for non-degenerate cases
    for var_name in var_names
        var_ = fetch_interpolate_transform(var_name, filename, z_scm)
        var_mean = !isnothing(tf) ? mean(var_[:, ti_index:tf_index], dims = 2) : var_[:, ti_index]
        append!(y, var_mean)
    end
    return y
end

function get_profile(m::ReferenceModel, filename::String; z_scm::Union{Vector{T}, T} = nothing) where {T}
    get_profile(m, filename, m.y_names, z_scm = z_scm)
end


function get_profile(
    m::ReferenceModel,
    filename::String,
    y_names::Vector{String};
    z_scm::Union{Vector{T}, T} = nothing,
) where {T}
    get_profile(filename, y_names, ti = get_t_start(m), tf = get_t_end(m), z_scm = z_scm)
end

"""
    get_time_covariance(m::ReferenceModel, var_names::Vector{String}; z_scm::Vector{FT}) where {FT <: Real}

Obtain the covariance matrix of a group of profiles, where the covariance
is obtained in time.
Inputs:
 - m            :: Reference model.
 - var_names    :: List of variable names to be included.
 - z_scm        :: If given, interpolates covariance matrix to this locations.
"""
function get_time_covariance(m::ReferenceModel, var_names::Vector{String}; z_scm::Vector{FT}) where {FT <: Real}
    filename = Σ_nc_file(m)
    t = nc_fetch(filename, "t")
    # Find closest interval in data
    ti_index = argmin(broadcast(abs, t .- get_t_start_Σ(m)))
    tf_index = argmin(broadcast(abs, t .- get_t_end_Σ(m)))
    N_samples = length(ti_index:tf_index)
    ts_vec = zeros(0, N_samples)
    num_outputs = length(var_names)
    pool_var = zeros(num_outputs)

    for (i, var_name) in enumerate(var_names)
        var_ = fetch_interpolate_transform(var_name, filename, z_scm)
        # Store pooled variance
        pool_var[i] = mean(var(var_[:, ti_index:tf_index], dims = 2)) + eps(FT) # vertically averaged time-variance of variable
        # Normalize timeseries
        ts_var_i = var_[:, ti_index:tf_index] ./ sqrt(pool_var[i])
        ts_vec = cat(ts_vec, ts_var_i, dims = 1)  # dims: (Nz*num_outputs, Nt)
    end
    cov_mat = cov(ts_vec, dims = 2)  # covariance, w/ samples across time dimension (t_inds).
    return cov_mat, pool_var
end

"""
    generate_ekp(
        ref_stats::ReferenceStatistics,
        process::Process,
        u::Union{Matrix{T}, T} = nothing;
        failure_handler::String = "ignore_failures",
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
    ekp = isnothing(u) ? EnsembleKalmanProcess(ref_stats.y, ref_stats.Γ, process, failure_handler_method = fh) :
        EnsembleKalmanProcess(u, ref_stats.y, ref_stats.Γ, process, failure_handler_method = fh)
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
            l2_reg_values = collect(values(l2_reg))
        end
        l2_reg_values = vcat(l2_reg_values...)

        # dict must be complete to preserve ordering
        @assert length(μ) == length(l2_reg_values) "Dictionary of regularizations l2_reg must include all parameters."

        # Augment exclusively with nonzero weights
        aug_indices = regularized_param_indices(l2_reg)
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

    ekp = isnothing(u) ? EnsembleKalmanProcess(y_aug, Γ_aug, process, failure_handler_method = fh) :
        EnsembleKalmanProcess(u, y_aug, Γ_aug, process, failure_handler_method = fh)
    if to_file
        jldsave(ekobj_path(outdir_path, 1); ekp)
    end
    return ekp
end

"Get indices of the parameters that are regularized for the augmented system"
function regularized_param_indices(l2_reg::Dict{String, Vector{FT}}) where {FT}
    # flatten l2_reg dict
    if any(1 .< [length(val) for val in collect(values(l2_reg))])
        _, l2_reg_values = flatten_config_dict(l2_reg)
    else
        l2_reg_values = collect(values(l2_reg))
    end
    l2_reg_values = vcat(l2_reg_values...)

    reg_indices = findall(x -> x > eps(FT), l2_reg_values)
    return reg_indices
end

end # module
