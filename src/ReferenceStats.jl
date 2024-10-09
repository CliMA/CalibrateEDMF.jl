module ReferenceStats

export ReferenceStatistics, pca_length, full_length, get_obs, get_profile, obs_PCA, pca, pca_inds, full_inds, get_ref_stats_kwargs

using SparseArrays
using Statistics
using Interpolations
using LinearAlgebra
using DocStringExtensions
using NaNStatistics # to handle when we have NaNs in our data

using ..AbstractTypes
using ..ReferenceModels
using ..ModelTypes
using ..LESUtils
using ..HelperFuncs
using ..DistributionUtils

import ..AbstractTypes: OptVec, OptReal

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
        time_shift::FT = 6 * 3600.0,
        model_errors::OptVec{T} = nothing,
        obs_var_scaling::Union{Dict, Nothing} = nothing,
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
 - `time_shift`               :: [LES last time - SCM start time (LES timeframe)] for `LES_driven_SCM` cases.
 - `model_errors`     :: Vector of model errors added to the internal variability noise, each containing
                            the model error per variable normalized by the pooled variable variance.
 - `obs_var_scaling`  :: Scaling factor for observational variance estimate
"""
Base.@kwdef struct ReferenceStatistics{FT <: Real, IT <: Integer}
    "Reference data, length: nSim * n_vars * n_zLevels (possibly reduced by PCA)"
    y::Vector{FT}
    "Data covariance matrix, dims: (y,y) (possibly reduced by PCA)"
    Γ::AbstractMatrix{FT}
    "Vector (length: `nSim`) of normalizing factors (length: `n_vars`)"
    norm_vec::Vector{Vector{FT}}
    "Vector (length: `nSim`) of PCA projection matrices with leading eigenvectors as columns"
    pca_vec::Vector{Union{AbstractMatrix{FT}, UniformScaling}}
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
        normalization_type::Symbol = :pooled_variance,
        characteristic_values::Union{Dict, Nothing} = nothing,
        variance_loss::FT = 0.1,
        tikhonov_noise::FT = 0.0,
        tikhonov_mode::String = "absolute",
        dim_scaling::Bool = false,
        time_shift::Union{FT, Vector{FT}} = 6 * 3600.0, # should be the reference time in the file, maybe this is 12*3600 for ours since that's the reference time? 
        model_errors::OptVec{T} = nothing,
        obs_var_scaling::Union{Dict, Nothing} = nothing,
        obs_var_additional_uncertainty_factor::Union{FT, Dict, Nothing} = nothing,
    ) where {FT <: Real, T}
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

        if isa(time_shift, Vector)
            @assert length(time_shift) == length(RM)
        else
            time_shift = repeat([time_shift], length(RM)) # expand to a vector if we're given a scalar
        end

        for (i, m) in enumerate(RM)
            model = m.case_name == "LES_driven_SCM" ? time_shift_reference_model(m, time_shift[i]) : m
            model = occursin("socrates", lowercase(m.case_name)) ? time_shift_reference_model(m, time_shift[i]) : m # if we use time shifts for SOCRATES cases ever, this will cover that - currently time_shift is 0.
            model_error = !isnothing(model_errors) ? model_errors[i] : nothing
            # Get (interpolated and pool-normalized) observations, get pool variance vector
            z_scm = get_z_rectified_obs(model) # should be rectified
            y_, y_var_, pool_var = get_obs(
                model,
                normalize;
                normalization_type = normalization_type,
                characteristic_values = characteristic_values,
                z_scm = z_scm, # trying here to allow for changing z_scm from what the reference was run with...
                model_error = model_error,
                obs_var_scaling = obs_var_scaling,
                obs_var_additional_uncertainty_factor = obs_var_additional_uncertainty_factor
            )
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
                push!(pca_vec, 1.0I(length(y_)))
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

    # # Framework for specifying mean and covariances a-priori: once we figure it out, we'll probably want something like:
    # function ReferenceStatistics(
    #     RM::Vector{ReferenceModel};
    #     perform_PCA::Bool = true,
    #     normalize::Bool = true,
    #     variance_loss::FT = 0.1,
    #     tikhonov_noise::FT = 0.0,
    #     tikhonov_mode::String = "absolute",
    #     dim_scaling::Bool = false,
    #     time_shift::Union{FT, Vector{FT}} = 6 * 3600.0, # should be the reference time in the file, maybe this is 12*3600 for ours since that's the reference time? 
    #     model_errors::OptVec{T} = nothing,
    #     reference_mean::Vector{Vector{Vector{{FT}}}}, # for each ReferenceModel, a mean profile for each variable
    #     reference_covariance::Vector{Vector{Vector{FT}}}, # for each ReferenceModel, a covariance matrix for each variable ReferenceModels{Vars{Zs}}
    # )

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
    get_obs(m::ReferenceModel, normalize; [z_scm])

Get observational mean `y` and empirical time covariance `Σ` for the [`ReferenceModel`](@ref) `m`.

Typically, the observations are fetched by specifying the `ReferenceModel`, which indicates
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
- `obs_var_scaling`  :: Scaling factor for observational variance estimate

# Keywords
- `z_scm`        :: If given, interpolate LES observations to given array of vertical levels.
- `model_error`  :: Model error per variable, added to the internal variability noise, and
                    normalized by the pooled variance of the variable.

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
    normalization_type::Symbol = :pooled_variance,
    characteristic_values::Union{Dict, Nothing} = nothing,
    z_scm::OptVec{FT} = nothing,
    model_error::OptVec{FT} = nothing,
    obs_var_scaling::Union{Dict, Nothing} = nothing,
    obs_var_additional_uncertainty_factor::Union{FT, Dict, Nothing} = nothing,
) where {FT <: Real}
    # time covariance
    Σ, pool_var = get_time_covariance(
        m,
        Σ_names,
        z_scm,
        normalize = normalize,
        normalization_type = normalization_type,
        characteristic_values = characteristic_values,
        model_error = model_error,
        obs_var_scaling = obs_var_scaling,
        obs_var_additional_uncertainty_factor = obs_var_additional_uncertainty_factor,
    )
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
    normalize::Bool;
    normalization_type::Symbol = :pooled_variance,
    characteristic_values::Union{Dict, Nothing} = nothing,
    z_scm::OptVec{FT},
    model_error::OptVec{FT} = nothing,
    obs_var_scaling::Union{Dict, Nothing} = nothing,
    obs_var_additional_uncertainty_factor::Union{FT, Dict, Nothing} = nothing,
) where {FT <: Real}
    y_names = isa(m.y_type, LES) ? get_les_names(m, y_nc_file(m)) : m.y_names
    Σ_names = isa(m.Σ_type, LES) ? get_les_names(m, Σ_nc_file(m)) : m.y_names
    get_obs(m, y_names, Σ_names, normalize, normalization_type = normalization_type, characteristic_values = characteristic_values, z_scm = z_scm, model_error = model_error, obs_var_scaling = obs_var_scaling, obs_var_additional_uncertainty_factor = obs_var_additional_uncertainty_factor)
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
function obs_PCA(y_mean::Vector{FT}, y_var::AbstractMatrix{FT}, allowed_var_loss::FT = 1.0e-1) where {FT <: Real}
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
        tf::OptReal = nothing,
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
 - `prof_ind`    :: Whether to return a boolean array indicating the variables that are profiles (i.e., not scalars). 

Outputs:

 - `y` :: Output vector used in the inverse problem, which concatenates the requested profiles.
"""
function get_profile(
    filename::String,
    y_names::Vector{String};
    ti::Real = 0.0,
    tf::OptReal = nothing,
    z_scm::OptVec{T} = nothing,
    prof_ind::Bool = false,
    penalized_value::FT = 1.0e5, # this was the default for training from Ignacio/Costa, but in other cases you may want other values, e.g. NaN
    verbose::Bool = true,
    has_time_dims::Union{Bool, Vector{Bool}} = true, # if time is not a dimension, e.g. it's a reference profile w/ only z as a dim
) where {T, FT}

    t = nc_fetch(filename, "t")
    dt = length(t) > 1 ? mean(diff(t)) : 0.0
    y = zeros(0)
    is_profile = Bool[]


    if isa(has_time_dims, Bool)
        has_time_dims = repeat([has_time_dims], length(y_names))
    end

    # Check that times are contained in simulation output
    Δt_start, ti_index = findmin(broadcast(abs, t .- ti))
    # If simulation does not contain values for ti or tf, return high value (penalization)
    if t[end] < ti
        if verbose
            @warn string(
                "Note: t_end < ti, which means that simulation stopped before reaching the requested t_start.",
                "Requested t_start = $ti s. However, the last time available is $(t[end]) s.",
                "Defaulting to penalized profiles...",
            )
        end
        for i in 1:length(y_names)
            var_ = isnothing(z_scm) ? get_height(filename) : z_scm
            append!(y, penalized_value * ones(length(var_[:])))
        end
        return prof_ind ? (y, repeat([true], length(y_names))) : y
    end
    if !isnothing(tf)
        Δt_end, tf_index = findmin(broadcast(abs, t .- tf))
        if t[end] < tf - dt
            if verbose
                @warn string(
                    "Note: t_end < tf - dt, which means that simulation stopped before reaching the requested t_end.",
                    "Requested t_end = $tf s. However, the last time available is $(t[end]) s.",
                    "Defaulting to penalized profiles...",
                )
            end
            for i in 1:length(y_names)
                var_ = isnothing(z_scm) ? get_height(filename) : z_scm
                append!(y, penalized_value * ones(length(var_[:])))
            end
            return prof_ind ? (y, repeat([true], length(y_names))) : y
        end
    end

    # Return time average for non-degenerate cases
    for (i_v, var_name) in enumerate(y_names)

        var_ = fetch_interpolate_transform(var_name, filename, z_scm)
        if has_time_dims[i_v]
            if ndims(var_) == 2
                var_mean = !isnothing(tf) ? mean(var_[:, ti_index:tf_index], dims = 2) : var_[:, ti_index]
                append!(is_profile, true)
            elseif ndims(var_) == 1
                var_mean = !isnothing(tf) ? mean(var_[ti_index:tf_index]) : var_[ti_index]
                append!(is_profile, false)
            end
        else
            var_mean = var_
            append!(is_profile, true) # it's just a z profile and we dont need to do anything else
        end
        
        append!(y, var_mean)
    end
    return prof_ind ? (y, is_profile) : y



end

function get_profile(m::ReferenceModel, filename::String; z_scm::OptVec{T} = nothing, prof_ind::Bool = false, penalized_value::FT = 1.0e5, verbose::Bool = true, has_time_dims::Union{Bool, Vector{Bool}} = true) where {T, FT}
    get_profile(m, filename, m.y_names, z_scm = z_scm, prof_ind = prof_ind, penalized_value = penalized_value, verbose = verbose, has_time_dims = has_time_dims) 
end

function get_profile(
    m::ReferenceModel,
    filename::String,
    y_names::Vector{String};
    z_scm::OptVec{T} = nothing,
    prof_ind::Bool = false,
    penalized_value::FT = 1.0e5, # this was the default for training from Ignacio/Costa, but in other cases you may want other values, e.g. NaN
    verbose::Bool = true,
    has_time_dims::Union{Bool, Vector{Bool}} = true
) where {T, FT}
    get_profile(filename, y_names, ti = get_t_start(m), tf = get_t_end(m), z_scm = z_scm, prof_ind = prof_ind, penalized_value = penalized_value, verbose = verbose, has_time_dims = has_time_dims)
end

"""
    get_time_covariance(
        m::ReferenceModel,
        y_names::Vector{String},
        z_scm::Vector{FT};
        normalize::Bool = true,
        model_error::OptVec{FT} = nothing,
        obs_var_scaling::Union{Dict, Nothing} = nothing,
    ) where {FT <: Real}

Obtain the covariance matrix of a group of profiles, where the covariance
is obtained in time.

Inputs:
 - `m`            :: Reference model.
 - `y_names`      :: List of variable names to be included.
 - `z_scm`        :: If given, interpolates covariance matrix to this locations.
 - `normalize`    :: Whether to normalize the time series with the pooled variance
        before computing the covariance, or not.
- `model_error`  :: Model error per variable, added to the internal variability noise, and
                    normalized by the pooled variance of the variable.
- `obs_var_scaling`  :: Scaling factor for observational variance estimate
"""
function get_time_covariance(
    m::ReferenceModel,
    y_names::Vector{String},
    z_scm::Vector{FT};
    normalize::Bool = true,
    normalization_type::Symbol = :pooled_variance,
    characteristic_values::Union{Dict, Nothing} = nothing,
    model_error::OptVec{FT} = nothing,
    obs_var_scaling::Union{Dict, Nothing} = nothing,
    obs_var_additional_uncertainty_factor::Union{FT, Dict, Nothing} = nothing, # if we do it by all vars
) where {FT <: Real}
    filename = Σ_nc_file(m)
    t = nc_fetch(filename, "t")
    # Find closest interval in data
    ti_index = argmin(broadcast(abs, t .- get_t_start_Σ(m)))
    tf_index = argmin(broadcast(abs, t .- get_t_end_Σ(m)))
    N_samples = length(ti_index:tf_index)
    ts_vec = zeros(0, N_samples)
    num_outputs = length(y_names)
    pool_var = zeros(num_outputs)
    model_error_expanded = Vector{FT}[]

    obs_var_additional_uncertainty_vectors = zeros(0, N_samples) # rows of data just like ts_vec

    for (i, var_name) in enumerate(y_names)

        if !isnothing(obs_var_scaling)
            if var_name == "s_flux_z"
                var_factor = get(obs_var_scaling, "total_flux_s", 1.0)
            elseif var_name in ("resolved_z_flux_qt", "qt_flux_z")
                var_factor = get(obs_var_scaling, "total_flux_qt", 1.0)
            else
                var_factor = get(obs_var_scaling, var_name, 1.0)
            end
        else
            var_factor = 1.0
        end

        var_ = fetch_interpolate_transform(var_name, filename, z_scm)

        if any(isnan, var_)
            @warn "NaNs in $var_name, switching to NaNStatistics"
            mean = NaNStatistics.nanmean
            var  = NaNStatistics.nanvar
        else
            mean = Statistics.mean # i think you haveto use this or else it fails...
            var  = Statistics.var
        end

        # @info characteristic_values

        if ndims(var_) == 2
            # Store pooled variance
            if normalization_type == :pooled_variance
                pool_var[i] = var_factor * mean(var(var_[:, ti_index:tf_index], dims = 2)) + eps(FT) # vertically averaged time-variance of variable
            elseif normalization_type == :pooled_nonzero_mean_to_value # convert the mean of non-zero values to 1
                # pool_var[i] = var_factor * mean_nonzero_elements(var_[:, ti_index:tf_index], dims=[1,2]) + eps(FT) # mean of nonzero elements of variable (should this be squared so the if block below isn't necessary?)
                mean_nonzero_elements_output = mean_nonzero_elements(var_[:, ti_index:tf_index])
                if iszero(mean_nonzero_elements_output) && !isnothing(characteristic_values) 
                    mean_nonzero_elements_output = get(characteristic_values, var_name, FT(0.0))
                    # @info "new mean_nonzero_elements_output: $mean_nonzero_elements_output"
                end
                pool_var[i] = var_factor * mean_nonzero_elements_output.^2 + eps(FT) # mean of nonzero elements of variable, squared bc they usually normalize by sqrt of this value...
            else
                throw(ArgumentError("Normalization type $normalization_type not recognized."))
            end
            # Normalize timeseries
            # if normalize
            #     if normalization_type == :pooled_variance
            #         ts_var_i = var_[:, ti_index:tf_index] ./ sqrt(pool_var[i]) 
            #     elseif normalization_type = :pooled_nonzero_mean_to_value
            #         ts_var_i = var_[:, ti_index:tf_index] ./ pool_var[i] # not sure if it should be sqrt or not...
            #     else
            #         throw(ArgumentError("Normalization type $normalization_type not recognized."))
            #     end
            # else
            #     ts_var_i = var_[:, ti_index:tf_index] # dims: (Nz, Nt)
            # end
            ts_var_i = normalize ? var_[:, ti_index:tf_index] ./ sqrt(pool_var[i]) : var_[:, ti_index:tf_index] # dims: (Nz, Nt)
        elseif ndims(var_) == 1
            # Store pooled variance
            if normalization_type == :pooled_variance
                pool_var[i] = var_factor * var(var_[ti_index:tf_index]) + eps(FT) # time-variance of variable
            elseif normalization_type == :pooled_nonzero_mean_to_value # convert the mean of non-zero values to 1
                # pool_var[i] = var_factor * mean_nonzero_elements(var_[ti_index:tf_index]) + eps(FT) # mean of nonzero elements of variable  (should this be squared so the if block below isn't necessary?)
                mean_nonzero_elements_output = mean_nonzero_elements(var_[ti_index:tf_index])
                if iszero(mean_nonzero_elements_output) && !isnothing(characteristic_values)
                    mean_nonzero_elements_output = get(characteristic_values, var_name, FT(0.0))
                end
                pool_var[i] = var_factor * mean_nonzero_elements_output.^2 + eps(FT)  # time-variance of variable, squared bc they usually normalize by sqrt of this value...
            else
                throw(ArgumentError("Normalization type $normalization_type not recognized."))
            end
            # Normalize timeseries
            # if normalize
            #     if normalization_type == :pooled_variance
            #         ts_var_i = Array(var_[ti_index:tf_index]') ./ sqrt(pool_var[i])
            #     elseif normalization_type = :pooled_nonzero_mean_to_value
            #         ts_var_i = Array(var_[ti_index:tf_index]') ./ pool_var[i] # not sure if it should be sqrt or not...
            #     else
            #         throw(ArgumentError("Normalization type $normalization_type not recognized."))
            #     end
            # else
            #     ts_var_i = Array(var_[ti_index:tf_index]') # dims: (1, Nt)
            # end
               
            ts_var_i =
                normalize ? Array(var_[ti_index:tf_index]') ./ sqrt(pool_var[i]) : Array(var_[ti_index:tf_index]') # dims: (1, Nt)
        else
            throw(ArgumentError("Variable `$var_name` has more than 2 dimensions, 1 or 2 were expected."))
        end
        ts_vec = cat(ts_vec, ts_var_i, dims = 1)  # final dims: (Nz*num_profiles + num_timeseries, Nt), build up row by row

        # Add structural model error
        if !isnothing(model_error)
            var_model_error = normalize ? model_error[i] : model_error[i] * pool_var[i]
            model_error_expanded = cat(model_error_expanded, repeat([var_model_error], size(ts_var_i, 1)), dims = 1)
            # probably should add something here to support other normalization types? or maybe it's fine bc, just not sure if it should be pool_var or squared or what
        end

        # add additional uncertainty
        if !isnothing(obs_var_additional_uncertainty_factor)
            if isa(obs_var_additional_uncertainty_factor, FT)
                obs_var_additional_uncertainty_factor_i = FT(obs_var_additional_uncertainty_factor)
            elseif isa(obs_var_additional_uncertainty_factor, Dict)
                obs_var_additional_uncertainty_factor_i = FT(get(obs_var_additional_uncertainty_factor, var_name, FT(0.0)))
            end
        else
            obs_var_additional_uncertainty_factor_i = FT(0.0) # default to not adding anything...
        end
        obs_var_additional_uncertainty_vectors = cat(obs_var_additional_uncertainty_vectors, ts_var_i .* obs_var_additional_uncertainty_factor_i, dims=1)
    end

    # set cov to be safe for NaNs
    if any(isnan,ts_vec)
        @warn "NaNs in ts_vec, switching to NaNStatistics"
        cov = NaNStatistics.nancov
    else
        cov = Statistics.cov # it keeps crashing if i dont put this, idk
    end

    cov_mat = cov(ts_vec, dims = 2)  # covariance, w/ samples across time dimension (t_inds).
    cov_mat = !isnothing(model_error) ? cov_mat + Diagonal(FT.(model_error_expanded)) : cov_mat

    # test doing the variance addition here scaled by the vector value mean across time
    # if !isnothing(obs_var_additional_uncertainty_factor)
    #     if isa(obs_var_additional_uncertainty_factor, FT)
    #         obs_var_additional_uncertainty_factor = FT(obs_var_additional_uncertainty_factor)
    #     elseif isa(obs_var_additional_uncertainty_factor, Dict)
    #         obs_var_additional_uncertainty_factor = FT(obs_var_additional_uncertainty_factor[var_name])
    #     end
    # else
    #     obs_var_additional_uncertainty_factor = FT(0.0) # default to not adding anything...
    # end

    # var_addition_vec = mean(ts_vec, dims = 2)[:] .* obs_var_additional_uncertainty_factor # mean across time, times the factor

    # if we wanted to do this by variable we would I guess want to construct this vector in the loop above?
    # cov_mat += Diagonal(obs_var_additional_uncertainty_vectors)
    cov_mat += Diagonal(mean(obs_var_additional_uncertainty_vectors, dims=2)[:]) # mean across time, make diagonal, add to cov matrix

    # If we've added uncertainty due to low variance, it doesn't change the fact that our covariances are near 0... should we add an option to add covariances? that get's dicey bcause we don't know magnitude or sign...
    # we could try to use the 3D data as an alternative, but if we're not let's consider at best theyre perfectly correlated and cov(aX, bY) = a*b*cov(X,Y) = a*b*var(X)

    # Although we may have trimmed our data, NaNs in the data could appear after interpolation, so we need to handle that.
    cov_mat = replace(cov_mat, NaN => 0.0) # just like off diagonal blocks are 0, if we get a NaN out even after trimming, we'll just set it to 0. (if you did not trim your data) | ps, i think we get the same output if we don't trim? trimming might be worse bc it could allow the interpolant to work in areas that otherwise might have given NaN => 0.0
    return cov_mat, pool_var
end

function get_ref_stats_kwargs(ref_config::Dict{Any, Any}, reg_config::Dict{Any, Any})
    model_errors = get_entry(ref_config, "model_errors", nothing)
    time_shift = get_entry(ref_config, "time_shift", 6.0 * 3600.0)
    perform_PCA = get_entry(reg_config, "perform_PCA", true)
    variance_loss = get_entry(reg_config, "variance_loss", 1.0e-2)
    normalize = get_entry(reg_config, "normalize", true)
    normalization_type = get_entry(reg_config, "normalization_type", :pooled_variance)
    characteristic_values = get_entry(ref_config, "characteristic_values", nothing)
    tikhonov_mode = get_entry(reg_config, "tikhonov_mode", "relative")
    tikhonov_noise = get_entry(reg_config, "tikhonov_noise", 1.0e-6)
    dim_scaling = get_entry(reg_config, "dim_scaling", true)
    obs_var_scaling = get_entry(reg_config, "obs_var_scaling", nothing)
    obs_var_additional_uncertainty_factor = get_entry(reg_config, "obs_var_additional_uncertainty_factor", nothing)
    return Dict(
        :perform_PCA => perform_PCA,
        :normalize => normalize,
        :normalization_type => normalization_type,
        :characteristic_values => characteristic_values,
        :variance_loss => variance_loss,
        :tikhonov_noise => tikhonov_noise,
        :tikhonov_mode => tikhonov_mode,
        :dim_scaling => dim_scaling,
        :model_errors => model_errors,
        :time_shift => time_shift,
        :obs_var_scaling => obs_var_scaling,
        :obs_var_additional_uncertainty_factor => obs_var_additional_uncertainty_factor,
    )
end

# """
# I was working on a new version of this function that would return a vector of ReferenceStatistics objects
# This was motivated by a desire to have time_shifts be a vector (I need different ones for obs/ERA in the event I ever calibrate them simultaneously)
# It works but ref_stats in Pipeline.jl gets sent to Diagnostics and further so it's maybe not a priority right now for me to chase all that down and do that yet...
# Would work with construct_reference_statistics() below
# """
# function get_ref_stats_kwargs(ref_config::Dict{Any, Any}, reg_config::Dict{Any, Any}) # we need to edit this to be more like get_ref_model_kwargs in ReferenceModels.jl, to allow for different time_shifts...
#     n_cases = length(ref_config["case_name"])
#     model_errors = expand_dict_entry(ref_config, "model_errors", n_cases; default = nothing)
#     time_shift = expand_dict_entry(ref_config, "time_shift", n_cases; default = 6.0 * 3600.0)
#     perform_PCA = expand_dict_entry(reg_config, "perform_PCA", n_cases; default = true)
#     variance_loss = expand_dict_entry(reg_config, "variance_loss", n_cases; default = 1.0e-2)
#     normalize = expand_dict_entry(reg_config, "normalize", n_cases; default = true)
#     tikhonov_mode = expand_dict_entry(reg_config, "tikhonov_mode", n_cases; default = "relative")
#     tikhonov_noise = expand_dict_entry(reg_config, "tikhonov_noise", n_cases; default = 1.0e-6)
#     dim_scaling = expand_dict_entry(reg_config, "dim_scaling", n_cases; default = true)

#     rs_kwargs =  Dict(
#         :perform_PCA => perform_PCA,
#         :normalize => normalize,
#         :variance_loss => variance_loss,
#         :tikhonov_noise => tikhonov_noise,
#         :tikhonov_mode => tikhonov_mode,
#         :dim_scaling => dim_scaling,
#         :model_errors => model_errors,
#         :time_shift => time_shift,
#     )

#     n_RS = n_cases # redundant but copying format from get_ref_model_kwargs() which is also redundant
#     for (k, v) in pairs(rm_kwargs)
#         if !(k in [:y_type, :Σ_type])
#             @assert length(v) == n_RM "Entry `$k` in the reference config file has length $(length(v)). Should have length $n_RM."
#         end
#     end
#     return rm_kwargs
# end

# function construct_reference_statistics(ref_models::Vector{ReferenceModel}; kwarg_ld::Dict)::Vector{ReferenceStatistics}
#     n_RS = length(kwarg_ld['time_shift']) # just pick a random one
#     ref_stats = Vector{ReferenceStatistics}()
#     for RS_i in 1:n_RS
#         kw = Dict(k => v[RS_i] for (k, v) in pairs(kwarg_ld))  # unpack dict

#         push!(
#             ref_stats,
#             ReferenceStatistics(
#                 ref_models[RS_i];
#                 perform_pca = get(kw, :perform_pca, nothing), 
#                 normalize = get(kw, :normalize, nothing),
#                 variance_loss = get(kw, :variance_loss, nothing),
#                 tikhonov_noise = get(kw, :tikhonov_noise, nothing),
#                 tikhonov_mode = kwarg_ld[:tikhonov_mode],
#                 dim_scaling = kwarg_ld[:dim_scaling],
#                 time_shift = get(kw, :time_shift, nothing),
#                 model_errors = get(kw, :model_errors, nothing),
#             ),
#         )
#     end
#     return ref_stats
# end

end # module
