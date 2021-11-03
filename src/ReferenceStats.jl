module ReferenceStats

using Statistics
using Interpolations
using LinearAlgebra
using Glob
using Random
using JLD2
# EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
using ..ReferenceModels
using ..ModelTypes
using ..LESUtils
include("helper_funcs.jl")

export ReferenceStatistics
export pca_length, full_length
export get_obs, get_profile, obs_PCA
export generate_ekp


"""
    struct ReferenceStatistics{FT <: Real}
    
A structure containing statistics from the reference model used to
define a well-posed inverse problem.
"""
Base.@kwdef struct ReferenceStatistics{FT <: Real}
    "Reference data, length: nSim * n_vars * n_zLevels(possibly reduced by PCA)"
    y::Vector{FT} # yt
    "Data covariance matrix, dims: (y,y) (possibly reduced by PCA)"
    Γ::Array{FT, 2}  # Γy
    "Vector (length: nSim) of normalizing factors (length: n_vars)"
    norm_vec::Vector{Array{FT, 1}}  # pool_var_list
    "Vector (length: nSim) of PCA projection matrices with leading eigenvectors as columns"
    pca_vec::Vector{Union{Array{FT, 2}, UniformScaling}}  # P_pca_list
    "Full reference data vector, length: nSim * n_vars * n_zLevels"
    y_full::Vector{FT}  # yt_big
    "Full covariance matrix, dims: (y,y)"
    Γ_full::Array{FT, 2}  # yt_var_big

    """
        ReferenceStatistics(
            RM::Vector{ReferenceModel},
            perform_PCA::Bool,
            normalize::Bool,
            FT::DataType = Float64;
            variance_loss::Float64 = 0.1,
            tikhonov_noise::Float64 = 0.0,
            tikhonov_mode::String = "absolute",
            dim_scaling::Bool = false,
            y_type::ModelType = LES(),
            Σ_type::ModelType = LES(),
        )

    Constructs the ReferenceStatistics defining the inverse problem.

    Inputs:
     - RM               :: Vector of `ReferenceModel`s
     - perform_PCA      :: Boolean specifying whether to perform PCA.
     - normalize        :: Boolean specifying whether to normalize the data.
     - variance_loss    :: Fraction of variance loss when performing PCA.
     - tikhonov_noise   :: Tikhonov regularization factor for covariance matrices.
     - tikhonov_mode    :: If "relative", tikhonov_noise is scaled by the minimum
        eigenvalue in the covariance matrix considered.
     - dim_scaling      :: Whether to scale covariance blocks by their size.
     - y_type           :: Type of reference mean data. Either LES() or SCM()
     - Σ_type           :: Type of reference covariance data. Either LES() or SCM()
    Outputs:
     - A ReferenceStatistics struct.
    """
    function ReferenceStatistics(
        RM::Vector{ReferenceModel},
        perform_PCA::Bool,
        normalize::Bool;
        variance_loss::FT = 0.1,
        tikhonov_noise::FT = 0.0,
        tikhonov_mode::String = "absolute",
        dim_scaling::Bool = false,
        y_type::ModelType = LES(),
        Σ_type::ModelType = LES(),
    ) where {FT <: Real}
        # Init arrays
        y = FT[]  # yt
        Γ_vec = Array{FT, 2}[]  # yt_var_list
        y_full = FT[]  # yt_big
        Γ_full_vec = Array{FT, 2}[]  # yt_var_list_big
        pca_vec = []  # P_pca_list
        norm_vec = Vector[]  # pool_var_list

        for m in RM
            # Get (interpolated and pool-normalized) observations, get pool variance vector
            z_scm = get_height(scm_dir(m))
            y_, y_var_, pool_var = get_obs(m, y_type, Σ_type, normalize, z_scm = z_scm)
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
        Γ_full = cat(Γ_full_vec..., dims = (1, 2))

        # Construct global observational covariance matrix, TSVD
        if tikhonov_mode == "relative"
            Γ_vec = map(x -> x + tikhonov_noise * minimum(diag(x)) * I, Γ_vec)
        else
            Γ_vec = map(x -> x + tikhonov_noise * I, Γ_vec)
        end
        # Scale by number of dimensions
        Γ_vec = dim_scaling ? map(x -> size(x, 1) * x, Γ_vec) : Γ_vec
        Γ = cat(Γ_vec..., dims = (1, 2))
        @assert isposdef(Γ)

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
    y = get_profile(m, y_dir(m), y_names, z_scm = z_scm)
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
    y_names = isa(y_type, LES) ? get_les_names(m) : m.y_names
    Σ_names = isa(Σ_type, LES) ? get_les_names(m) : m.y_names
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
function obs_PCA(y_mean, y_var, allowed_var_loss = 1.0e-1)
    eig = eigen(y_var)
    eigvals, eigvecs = eig # eigvecs is matrix with eigvecs as cols
    # Get index of leading eigenvalues, eigvals are ordered from low to high in julia
    # This expression recovers 1 extra eigenvalue compared to threshold
    leading_eigs = findall(<(1.0 - allowed_var_loss), -cumsum(eigvals) / sum(eigvals) .+ 1)
    P_pca = eigvecs[:, leading_eigs]
    λ_pca = eigvals[leading_eigs]
    # Check correct PCA projection
    @assert Diagonal(λ_pca) ≈ P_pca' * y_var * P_pca
    # Project mean
    y_pca = P_pca' * y_mean
    y_var_pca = Diagonal(λ_pca)
    return y_pca, y_var_pca, P_pca
end


function get_profile(m::ReferenceModel, sim_dir::String; z_scm::Union{Vector{Float64}, Nothing} = nothing)
    get_profile(m, sim_dir, m.y_names, z_scm = z_scm)
end


function get_profile(
    m::ReferenceModel,
    sim_dir::String,
    y_names::Vector{String};
    z_scm::Union{Vector{Float64}, Nothing} = nothing,
)
    get_profile(sim_dir, y_names, ti = get_t_start(m), tf = get_t_end(m), z_scm = z_scm)
end

"""
    get_profile(
        sim_dir::String,
        var_names::Vector{String};
        ti::Real = 0.0,
        tf = nothing,
        z_scm::Union{Vector{Float64}, Nothing} = nothing,
    )

Get profiles for variables var_names, interpolated to
z_scm (if given), and concatenated into a single output vector.

Inputs:
 - sim_dir  :: Simulation output directory.
 - var_names   :: Names of variables to be retrieved.
 - z_scm :: If given, interpolate LES observations to given levels.
Outputs:
 - y :: Output vector used in the inverse problem, which concatenates the
   requested profiles. 
"""
function get_profile(
    sim_dir::String,
    var_names::Vector{String};
    ti::Real = 0.0,
    tf::Union{Real, Nothing} = nothing,
    z_scm::Union{Vector{Float64}, Nothing} = nothing,
)

    t = nc_fetch(sim_dir, "timeseries", "t")
    dt = length(t) > 1 ? abs(t[2] - t[1]) : 0.0
    y = zeros(0)

    # Check that times are contained in simulation output
    Δt_start, ti_index = findmin(broadcast(abs, t .- ti))
    # If simulation does not contain values for ti or tf, return high value (penalization)
    if Δt_start > dt
        @warn string(
            "Note: Δt_start > dt, which means that simulation stopped before reaching the requested t_start.",
            "Requested t_start = $ti s. However, the last time available is $(t[end]) s.",
            "Defaulting to penalized profiles...",
        )
        for i in 1:length(var_names)
            var_ = get_height(sim_dir)
            append!(y, 1.0e5 * ones(length(var_[:])))
        end
        return y
    end
    if !isnothing(tf)
        Δt_end, tf_index = findmin(broadcast(abs, t .- tf))
        if Δt_end > dt
            @warn string(
                "Note: Δt_end > dt, which means that simulation stopped before reaching the requested t_end.",
                "Requested t_end = $tf s. However, the last time available is $(t[end]) s.",
                "Defaulting to penalized profiles...",
            )
            for i in 1:length(var_names)
                var_ = get_height(sim_dir)
                append!(y, 1.0e5 * ones(length(var_[:])))
            end
            return y
        end
    end

    # Return time average for non-degenerate cases
    for var_name in var_names
        var_ = fetch_interpolate_transform(var_name, sim_dir, z_scm)
        var_mean = !isnothing(tf) ? mean(var_[:, ti_index:tf_index], dims = 2) : var_[:, ti_index]
        append!(y, var_mean)
    end
    return y
end

"""
    get_time_covariance(
        m::ReferenceModel,
        var_names::Vector{String};
        z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
    )

Obtain the covariance matrix of a group of profiles, where the covariance
is obtained in time.
Inputs:
 - m            :: Reference model.
 - var_names    :: List of variable names to be included.
 - z_scm        :: If given, interpolates covariance matrix to this locations.
"""
function get_time_covariance(
    m::ReferenceModel,
    var_names::Vector{String};
    z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
)
    sim_dir = Σ_dir(m)
    t = nc_fetch(sim_dir, "timeseries", "t")
    # Find closest interval in data
    ti_index = argmin(broadcast(abs, t .- get_t_start_Σ(m)))
    tf_index = argmin(broadcast(abs, t .- get_t_end_Σ(m)))
    ts_vec = zeros(0, length(ti_index:tf_index))
    num_outputs = length(var_names)
    pool_var = zeros(num_outputs)

    for (i, var_name) in enumerate(var_names)
        var_ = fetch_interpolate_transform(var_name, sim_dir, z_scm)
        # Store pooled variance
        pool_var[i] = mean(var(var_[:, ti_index:tf_index], dims = 2))  # vertically averaged time-variance of variable
        # Normalize timeseries
        ts_var_i = var_[:, ti_index:tf_index] ./ sqrt(pool_var[i])
        ts_vec = cat(ts_vec, ts_var_i, dims = 1)  # dims: (Nz*num_outputs, Nt)
    end
    cov_mat = cov(ts_vec, dims = 2)  # covariance, w/ samples across time dimension (t_inds).
    return cov_mat, pool_var
end

"""
    generate_ekp(
        u::Array{Float64, 2},
        ref_stats::ReferenceStatistics,
        algo;
        outdir_path::String = pwd(),
        to_file::Bool = true,
    )

Generates, and possible writes to file, an EnsembleKalmanProcess
from a parameter ensemble and reference statistics.

Inputs:
 - u :: An ensemble of parameter vectors.
 - ref_stats :: ReferenceStatistics defining the inverse problem.
 - algo :: Type of EnsembleKalmanProcess algorithm used to evolve the ensemble.
 - outdir_path :: Output path.
 - to_file :: Whether to write the serialized prior to a JLD2 file.

Output:
 - The generated EnsembleKalmanProcess.
"""
function generate_ekp(
    u::Array{Float64, 2},
    ref_stats::ReferenceStatistics,
    algo;
    outdir_path::String = pwd(),
    to_file::Bool = true,
)
    ekp_obj = EnsembleKalmanProcess(u, ref_stats.y, ref_stats.Γ, algo)
    if to_file
        ekp = serialize_struct(ekp_obj)
        jldsave(ekobj_path(outdir_path, 1); ekp)
    end
    return ekp_obj
end

function generate_ekp(
    ref_stats::ReferenceStatistics,
    algo::Unscented{FT, IT};
    outdir_path::String = pwd(),
    to_file::Bool = true,
) where {FT <: Real, IT <: Integer}
    ekp_obj = EnsembleKalmanProcess(ref_stats.y, ref_stats.Γ, algo)
    if to_file
        ekp = serialize_struct(ekp_obj)
        jldsave(ekobj_path(outdir_path, 1); ekp)
    end
    return ekp_obj
end

end # module
