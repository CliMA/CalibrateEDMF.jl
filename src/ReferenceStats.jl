module ReferenceStats

using Statistics
using Interpolations
using LinearAlgebra
using Glob
using Random
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributionStorage
using ..ReferenceModels
include(joinpath(@__DIR__, "helper_funcs.jl"))

export ReferenceStatistics
export pca_length, full_length


"""
    struct ReferenceStatistics
    
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
            model_type::Symbol,
            perform_PCA::Bool,
            normalize::Bool,
            FT::DataType = Float64;
            variance_loss::Float64 = 0.1,
            tikhonov_noise::Float64 = 0.0,
            tikhonov_mode::String = "absolute",
            Γ_scaling::Float64 = 1.0,
        )

    Constructs the ReferenceStatistics defining the inverse problem.

    Inputs:
     - RM               :: Vector of `ReferenceModel`s
     - model_type       :: Type of the reference model, either :les or :scm.
     - perform_PCA      :: Boolean specifying whether to perform PCA.
     - normalize        :: Boolean specifying whether to normalize the data.
     - variance_loss    :: Fraction of variance loss when performing PCA.
     - tikhonov_noise   :: Tikhonov regularization factor for covariance matrices.
     - tikhonov_mode    :: If "relative", tikhonov_noise is scaled by the minimum
        eigenvalue in the covariance matrix considered.
     - dim_scaling      :: Whether to scale covariance blocks by their size.
    Outputs:
     - A ReferenceStatistics struct.
    """
    function ReferenceStatistics(
        RM::Vector{ReferenceModel},
        model_type::Symbol,
        perform_PCA::Bool,
        normalize::Bool,
        FT::DataType = Float64;
        variance_loss::Float64 = 0.1,
        tikhonov_noise::Float64 = 0.0,
        tikhonov_mode::String = "absolute",
        dim_scaling::Bool = false,
    )
        # Init arrays
        y = FT[]  # yt
        Γ_vec = Array{FT, 2}[]  # yt_var_list
        y_full = FT[]  # yt_big
        Γ_full_vec = Array{FT, 2}[]  # yt_var_list_big
        pca_vec = []  # P_pca_list
        norm_vec = Vector[]  # pool_var_list

        for m in RM
            # Get (interpolated and pool-normalized) observations, get pool variance vector
            y_, y_var_, pool_var = get_obs(model_type, m, z_scm = get_height(scm_dir(m)), normalize)
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


end # module
