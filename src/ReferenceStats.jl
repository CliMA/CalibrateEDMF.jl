module ReferenceStats

using SparseArrays
using Statistics
using Interpolations
using LinearAlgebra
using DocStringExtensions

using ..ReferenceModels
using ..ModelTypes
using ..LESUtils
using ..HelperFuncs
using ..DistributionUtils

export ReferenceStatistics
export pca_length, full_length
export get_obs, get_profile, obs_PCA, pca
export pca_inds, full_inds


"""
    ReferenceStatistics{FT <: Real, IT <: Integer}

A structure containing statistics from the reference model used to
define a well-posed inverse problem.

# Fields

$(TYPEDFIELDS)

# Constructors

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
 - `RM`               :: Vector of `ReferenceModel`s.
 - `perform_PCA`      :: Boolean specifying whether to perform PCA.
 - `normalize`        :: Boolean specifying whether to normalize the data.
 - `variance_loss`    :: Fraction of variance loss when performing PCA.
 - `tikhonov_noise`   :: Tikhonov regularization factor for covariance matrices.
 - `tikhonov_mode`    :: If "relative", tikhonov_noise is scaled by the maximum
    eigenvalue in the covariance matrix considered, having the interpretation of
    the inverse of the desired condition number. This value is enforced to be
    larger than the sqrt of the machine precision for stability.
 - `dim_scaling`      :: Whether to scale covariance blocks by their size.
 - `y_type`           :: Type of reference mean data. Either LES() or SCM().
 - `Σ_type`           :: Type of reference covariance data. Either LES() or SCM().
 - `Δt`               :: [LES last time - SCM start time (LES timeframe)] for `LES_driven_SCM` cases.
"""
Base.@kwdef struct ReferenceStatistics{FT <: Real, IT <: Integer}
    "Reference data, length: nSim * n_vars * n_zLevels (possibly reduced by PCA)"
    y::Vector{FT}
    "Data covariance matrix, dims: (y,y) (possibly reduced by PCA)"
    Γ::Matrix{FT}
    "Vector (length: `nSim`) of normalizing factors (length: `n_vars`)"
    norm_vec::Vector{Vector{FT}}
    "Vector (length: `nSim`) of PCA projection matrices with leading eigenvectors as columns"
    pca_vec::Vector{Union{Matrix{FT}, UniformScaling}}
    "Full reference data vector, length: `nSim * n_vars * n_zLevels`"
    y_full::Vector{FT}
    "Full covariance matrix, dims: (y,y)"
    Γ_full::SparseMatrixCSC{FT, Int64}
    "Degrees of freedom per case (possibly reduced by PCA)"
    ndof_case::Vector{IT}
    "Full degrees of freedom per case: `zdof * n_vars`"
    ndof_full_case::Vector{IT}
    "Vertical degrees of freedom in profiles per case"
    zdof::Vector{IT}
    # Constructors

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
        IT = Int64
        # Init arrays
        y = FT[]
        Γ_vec = Matrix{FT}[]
        y_full = FT[]
        Γ_full_vec = Matrix{FT}[]
        pca_vec = []
        norm_vec = Vector[]
        ndof_case = IT[]
        ndof_full_case = IT[]
        zdof = IT[]

        for m in RM
            model = m.case_name == "LES_driven_SCM" ? time_shift_reference_model(m, Δt) : m
            # Get (interpolated and pool-normalized) observations, get pool variance vector
            y_, y_var_, pool_var = get_obs(model, y_type, Σ_type, normalize, z_scm = get_z_obs(model))
            push!(norm_vec, pool_var)
            if perform_PCA
                y_pca, y_var_pca, P_pca = obs_PCA(y_, y_var_, variance_loss)
                append!(y, y_pca)
                push!(Γ_vec, y_var_pca)
                push!(pca_vec, P_pca)
                push!(ndof_case, length(y_pca))
            else
                append!(y, y_)
                push!(Γ_vec, y_var_)
                push!(pca_vec, 1.0I)
                push!(ndof_case, length(y_))
            end
            # Save full dimensionality (normalized) output for error computation
            append!(y_full, y_)
            push!(Γ_full_vec, y_var_)
            push!(ndof_full_case, length(y_))
            push!(zdof, length(get_z_obs(model)))
        end

        # Construct global observational covariance matrix, original space
        Γ_full = sparse(cat(Γ_full_vec..., dims = (1, 2)))

        # Scale by number of dimensions (averaging loss per dimension)
        Γ_vec = dim_scaling ? length(Γ_vec) .* map(x -> size(x, 1) * x, Γ_vec) : Γ_vec
        # Construct global observational covariance matrix, PCA
        Γ = cat(Γ_vec..., dims = (1, 2))
        # Condition global covariance matrix, PCA
        if tikhonov_mode == "relative"
            tikhonov_noise = max(tikhonov_noise, 10 * sqrt(eps(FT)))
            Γ = Γ + tikhonov_noise * eigmax(Γ) * I
        else
            Γ = Γ + tikhonov_noise * I
        end

        @assert isposdef(Γ) "Covariance matrix Γ is ill-conditioned, consider regularization."

        return new{FT, IT}(y, Γ, norm_vec, pca_vec, y_full, Γ_full, ndof_case, ndof_full_case, zdof)
    end

end

"Returns dimensionality of the ReferenceStatistics in low-dimensional latent space"
pca_length(RS::ReferenceStatistics) = length(RS.y)

"Returns full dimensionality of the ReferenceStatistics, before latent space encoding"
full_length(RS::ReferenceStatistics) = length(RS.y_full)
pca_length(RS::ReferenceStatistics, case_ind) = RS.ndof_case[case_ind]
full_length(RS::ReferenceStatistics, case_ind) = RS.ndof_full_case[case_ind]

"""Fetch the dof indices (possibly reduced by PCA) for the a case, specified by its in config["reference"]["case_name"]."""
pca_inds(RS::ReferenceStatistics, case_ind) = dof_inds(RS.ndof_case, case_ind)
"""Fetch the dof indices for the a case, specified by its in config["reference"]["case_name"]."""
full_inds(RS::ReferenceStatistics, case_ind) = dof_inds(RS.ndof_full_case, case_ind)

"""Given a vector of `ndofs` for some ordered collection, fetch the dof indices for one of the elements by its index in the ordered collection."""
function dof_inds(ndofs::Vector{IT}, ind::IT) where {IT <: Integer}
    !(1 ≤ ind ≤ length(ndofs)) && throw(ArgumentError("Index `$ind` must be between 1 and $(length(ndofs))."))
    start_ind = (1, 1 .+ cumsum(ndofs)...)[ind]
    stop_ind = cumsum(ndofs)[ind]
    start_ind:stop_ind
end

"""
    get_obs(m::ReferenceModel, y_names, Σ_names, normalize; [z_scm])
    get_obs(m::ReferenceModel, y_type, Σ_type, normalize; [z_scm])

Get observational mean `y` and empirical time covariance `Σ` for the [`ReferenceModel`](@ref) `m`.

Typically, the observations are fetched by specifying `y_type` and `Σ_type`, which indicate
if the data is generated by the [`SCM`](@ref) or [`LES`](@ref).
Alternatively, vectors of variable names, `y_names`, `Σ_names`, can be specified directly.
Note: `Σ_names` may be different than `y_names` if there are LES/SCM name discrepancies.

The keyword `normalize` specifies whether observations are to be normalized with respect to the 
per-quantity pooled variance or not. See [`normalize_profile`](@ref) for details.
The normalization vector is return along with `y` and `Σ`.

If `z_scm` is given, interpolate observations to the given levels.

# Arguments
- `m`            :: A `ReferenceModel`](@ref)
- `y_names`      :: Names of observed fields from the [`ReferenceModel`](@ref) `m`.
- `Σ_names`      :: Names of fields used to construct covariances, may be different than `y_names`
    if there are LES/SCM name discrepancies.
- `normalize`    :: Whether to normalize the observations.
- `y_type`       :: `ModelType` used to construct observations, `::LES` or `::SCM`.
- `y_type`       :: `ModelType` used to construct noise covariances, `::LES` or `::SCM`.

# Keywords
- `z_scm`        :: If given, interpolate LES observations to given array of vertical levels.

# Returns
- `y::Vector`           :: Mean of observations `y`, possibly interpolated to `z_scm` levels.
- `Σ::Matrix`           :: Observational covariance matrix `Σ`, possibly pool-normalized.
- `norm_vec::Vector`    :: Vertically averaged time-variance, one entry for each variable
"""
function get_obs(
    m::ReferenceModel,
    y_names::Vector{String},
    Σ_names::Vector{String},
    normalize::Bool;
    z_scm::Union{Vector{FT}, Nothing} = nothing,
) where {FT <: Real}
    # time covariance
    Σ, pool_var = get_time_covariance(m, Σ_names, z_scm)
    # normalization
    norm_vec = normalize ? pool_var : ones(size(pool_var))
    # Get true observables
    y, prof_indices = get_profile(m, y_nc_file(m), y_names, z_scm = z_scm, prof_ind = true)
    # normalize
    y = normalize_profile(y, norm_vec, length(z_scm), prof_indices)
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
the variance `y_var`. Only eigenvectors with eigenvalues that contribute
to the leading 1-`allowed_var_loss` variance are retained.
Inputs:

 - `y_mean`           :: Mean of the observations.
 - `y_var`            :: Variance of the observations.
 - `allowed_var_loss` :: Maximum variance loss allowed.

Outputs:

 - `y_pca`            :: Projection of `y_mean` onto principal subspace spanned by eigenvectors.
 - `y_var_pca`        :: Projection of `y_var` on principal subspace.
 - `P_pca`            :: Projection matrix onto principal subspace, with leading eigenvectors as columns.
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
 - `covmat`           :: Variance of the observations.
 - `allowed_var_loss` :: Maximum variance loss allowed.
Outputs:
 - `λ_pca`            :: Principal eigenvalues, ordered in increasing value order.
 - `P_pca`            :: Projection matrix onto principal subspace, with leading eigenvectors as columns.
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
        y_names::Vector{String};
        ti::Real = 0.0,
        tf::Union{Real, Nothing} = nothing,
        z_scm::Union{Vector{T}, T} = nothing,
        prof_ind::Bool = false,
    ) where {T}
    get_profile(
        m::ReferenceModel,
        filename::String;
        z_scm::Union{Vector{T}, T} = nothing,
        prof_ind::Bool = false,
    ) where {T}
    get_profile(
        m::ReferenceModel,
        filename::String,
        y_names::Vector{String};
        z_scm::Union{Vector{T}, T} = nothing,
        prof_ind::Bool = false,
    ) where {T}

Get time-averaged profiles for variables `y_names`, interpolated to
`z_scm` (if given), and concatenated into a single output vector.

Inputs:

 - `filename`    :: nc filename
 - `y_names`     :: Names of variables to be retrieved.
 - `ti`          :: Initial time of averaging window.
 - `tf`          :: Final time of averaging window.
 - `z_scm`       :: If given, interpolate LES observations to given levels.
 - `m`           :: ReferenceModel from which to fetch profiles, implicitly defines `ti` and `tf`.
 - `prof_ind`    :: Whether to return a boolean array indicated the variables that are profiles (i.e., not scalars). 

Outputs:

 - `y` :: Output vector used in the inverse problem, which concatenates the requested profiles.
"""
function get_profile(
    filename::String,
    y_names::Vector{String};
    ti::Real = 0.0,
    tf::Union{Real, Nothing} = nothing,
    z_scm::Union{Vector{T}, T} = nothing,
    prof_ind::Bool = false,
) where {T}

    t = nc_fetch(filename, "t")
    dt = length(t) > 1 ? mean(diff(t)) : 0.0
    y = zeros(0)
    is_profile = Bool[]

    # Check that times are contained in simulation output
    Δt_start, ti_index = findmin(broadcast(abs, t .- ti))
    # If simulation does not contain values for ti or tf, return high value (penalization)
    if t[end] < ti
        @warn string(
            "Note: t_end < ti, which means that simulation stopped before reaching the requested t_start.",
            "Requested t_start = $ti s. However, the last time available is $(t[end]) s.",
            "Defaulting to penalized profiles...",
        )
        for i in 1:length(y_names)
            var_ = isnothing(z_scm) ? get_height(filename) : z_scm
            append!(y, 1.0e5 * ones(length(var_[:])))
        end
        return prof_ind ? (y, repeat([true], length(y_names))) : y
    end
    if !isnothing(tf)
        Δt_end, tf_index = findmin(broadcast(abs, t .- tf))
        if t[end] < tf - dt
            @warn string(
                "Note: t_end < tf - dt, which means that simulation stopped before reaching the requested t_end.",
                "Requested t_end = $tf s. However, the last time available is $(t[end]) s.",
                "Defaulting to penalized profiles...",
            )
            for i in 1:length(y_names)
                var_ = isnothing(z_scm) ? get_height(filename) : z_scm
                append!(y, 1.0e5 * ones(length(var_[:])))
            end
            return prof_ind ? (y, repeat([true], length(y_names))) : y
        end
    end

    # Return time average for non-degenerate cases
    for var_name in y_names
        var_ = fetch_interpolate_transform(var_name, filename, z_scm)
        if ndims(var_) == 2
            var_mean = !isnothing(tf) ? mean(var_[:, ti_index:tf_index], dims = 2) : var_[:, ti_index]
            append!(is_profile, true)
        elseif ndims(var_) == 1
            var_mean = !isnothing(tf) ? mean(var_[ti_index:tf_index]) : var_[ti_index]
            append!(is_profile, false)
        end
        append!(y, var_mean)
    end
    return prof_ind ? (y, is_profile) : y
end

function get_profile(
    m::ReferenceModel,
    filename::String;
    z_scm::Union{Vector{T}, T} = nothing,
    prof_ind::Bool = false,
) where {T}
    get_profile(m, filename, m.y_names, z_scm = z_scm, prof_ind = prof_ind)
end

function get_profile(
    m::ReferenceModel,
    filename::String,
    y_names::Vector{String};
    z_scm::Union{Vector{T}, T} = nothing,
    prof_ind::Bool = false,
) where {T}
    get_profile(filename, y_names, ti = get_t_start(m), tf = get_t_end(m), z_scm = z_scm, prof_ind = prof_ind)
end

"""
    get_time_covariance(m::ReferenceModel, y_names::Vector{String}, z_scm::Vector{FT}) where {FT <: Real}

Obtain the covariance matrix of a group of profiles, where the covariance
is obtained in time.

Inputs:
 - `m`            :: Reference model.
 - `y_names`    :: List of variable names to be included.
 - `z_scm`        :: If given, interpolates covariance matrix to this locations.
"""
function get_time_covariance(m::ReferenceModel, y_names::Vector{String}, z_scm::Vector{FT}) where {FT <: Real}
    filename = Σ_nc_file(m)
    t = nc_fetch(filename, "t")
    # Find closest interval in data
    ti_index = argmin(broadcast(abs, t .- get_t_start_Σ(m)))
    tf_index = argmin(broadcast(abs, t .- get_t_end_Σ(m)))
    N_samples = length(ti_index:tf_index)
    ts_vec = zeros(0, N_samples)
    num_outputs = length(y_names)
    pool_var = zeros(num_outputs)

    for (i, var_name) in enumerate(y_names)
        var_ = fetch_interpolate_transform(var_name, filename, z_scm)
        if ndims(var_) == 2
            # Store pooled variance
            pool_var[i] = mean(var(var_[:, ti_index:tf_index], dims = 2)) + eps(FT) # vertically averaged time-variance of variable
            # Normalize timeseries
            ts_var_i = var_[:, ti_index:tf_index] ./ sqrt(pool_var[i]) # dims: (Nz, Nt)
        elseif ndims(var_) == 1
            # Store pooled variance
            pool_var[i] = var(var_[ti_index:tf_index]) + eps(FT) # time-variance of variable
            # Normalize timeseries
            ts_var_i = Array(var_[ti_index:tf_index]') ./ sqrt(pool_var[i]) # dims: (1, Nt)
        else
            throw(ArgumentError("Variable `$var_name` has more than 2 dimensions, 1 or 2 were expected."))
        end
        ts_vec = cat(ts_vec, ts_var_i, dims = 1)  # final dims: (Nz*num_profiles + num_timeseries, Nt)
    end
    cov_mat = cov(ts_vec, dims = 2)  # covariance, w/ samples across time dimension (t_inds).
    return cov_mat, pool_var
end

end # module
