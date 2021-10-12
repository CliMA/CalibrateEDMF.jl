"""Generic utils."""

using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob
using JSON
using Random
using CalibrateEDMF.ReferenceModels
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributionStorage
# TurbulenceConvection.jl
using TurbulenceConvection
tc_dir = dirname(dirname(pathof(TurbulenceConvection)));
include(joinpath(tc_dir, "integration_tests", "utils", "main.jl"))


"""
    get_obs(
        obs_type::Symbol,
        m::ReferenceModel;
        z_scm::Union{Vector{FT}, Nothing} = nothing,
    ) where FT<:Real

Get observations for variables y_names, interpolated to
z_scm (if given), and possibly normalized with respect to the pooled variance.

Inputs:
 - obs_type     :: Either :les or :scm
 - m            :: Reference model
 - z_scm :: If given, interpolate LES observations to given levels.
Outputs:
 - y_ :: Mean of observations, possibly interpolated to z_scm levels.
 - y_tvar :: Observational covariance matrix, possibly pool-normalized.
 - pool_var :: Vector of vertically averaged time-variance, one entry for each variable
"""
function get_obs(
    obs_type::Symbol,
    m::ReferenceModel,
    normalize::Bool;
    z_scm::Union{Vector{FT}, Nothing} = nothing,
) where {FT <: Real}
    les_names = get_les_names(m.y_names, m.les_dir)

    # True observables from SCM or LES depending on `obs_type` flag
    y_names, sim_dir = if obs_type == :scm
        m.y_names, m.scm_dir
    elseif obs_type == :les
        les_names, m.les_dir
    else
        error("Unknown observation type $obs_type")
    end

    # For now, we always use LES to construct covariance matrix
    y_tvar, pool_var = get_time_covariance(m, m.les_dir, les_names, z_scm = z_scm)

    norm_vec = if normalize
        pool_var
    else
        ones(size(pool_var))
    end

    # Get true observables
    y_highres = get_profile(m, sim_dir, y_names)
    # normalize
    y_highres = normalize_profile(y_highres, num_vars(m), norm_vec)

    if !isnothing(z_scm)
        y_ = zeros(0)
        z_les = get_height(sim_dir)
        num_outputs = Integer(length(y_highres) / length(z_les))
        for i in 1:num_outputs
            y_itp =
                interpolate((z_les,), y_highres[(1 + length(z_les) * (i - 1)):(i * length(z_les))], Gridded(Linear()))
            append!(y_, y_itp(z_scm))
        end
    else
        y_ = y_highres
    end
    return y_, y_tvar, norm_vec
end


"""
    obs_PCA(y_mean, y_var, allowed_var_loss = 1.0e-1)

Perform dimensionality reduction using principal component analysis on
the variance y_var. Only eigenvectors with eigenvalues that contribute
to the leading 1-allowed_var_loss variance are retained.
Inputs:
 - y_mean :: Mean of the observations.
 - y_var :: Variance of the observations.
 - allowed_var_loss :: Maximum variance loss allowed.
Outputs:
 - y_pca :: Projection of y_mean onto principal subspace spanned by eigenvectors.
 - y_var_pca :: Projection of y_var on principal subspace.
 - P_pca :: Projection matrix onto principal subspace, with leading eigenvectors as columns.
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


function get_profile(m::ReferenceModel, sim_dir::String)
    get_profile(m, sim_dir, m.y_names)
end


function get_profile(m::ReferenceModel, sim_dir::String, y_names::Vector{String})
    get_profile(sim_dir, y_names, ti = m.t_start, tf = m.t_end)
end


function get_profile(sim_dir::String, var_name::Vector{String}; ti::Real = 0.0, tf = nothing)

    t = nc_fetch(sim_dir, "timeseries", "t")
    dt = length(t) > 1 ? abs(t[2] - t[1]) : 0.0
    # Check that times are contained in simulation output
    ti_diff, ti_index = findmin(broadcast(abs, t .- ti))
    if !isnothing(tf)
        tf_diff, tf_index = findmin(broadcast(abs, t .- tf))
    end
    prof_vec = zeros(0)
    # If simulation does not contain values for ti or tf, return high value
    if ti_diff > dt
        println("ti_diff > dt ", "ti_diff = ", ti_diff, "dt = ", dt, "ti = ", ti, "t[1] = ", t[1], "t[end] = ", t[end])
        for i in 1:length(var_name)
            var_ = get_height(sim_dir)
            append!(prof_vec, 1.0e5 * ones(length(var_[:])))
        end
    else
        for i in 1:length(var_name)
            if occursin("horizontal_vel", var_name[i])
                u_ = nc_fetch(sim_dir, "profiles", "u_mean")
                v_ = nc_fetch(sim_dir, "profiles", "v_mean")
                var_ = sqrt.(u_ .^ 2 + v_ .^ 2)
            else
                var_ = nc_fetch(sim_dir, "profiles", var_name[i])
                # LES vertical fluxes are per volume, not mass
                if occursin("resolved_z_flux", var_name[i])
                    rho_half = nc_fetch(sim_dir, "reference", "rho0_half")
                    var_ = var_ .* rho_half
                end
            end
            if !isnothing(tf)
                append!(prof_vec, mean(var_[:, ti_index:tf_index], dims = 2))
            else
                append!(prof_vec, var_[:, ti_index])
            end
        end
    end
    return prof_vec
end


"""
    get_height(sim_dir::String; get_faces::Bool = false)

Returns the vertical cell centers or faces of the given configuration.

Inputs:
 - sim_dir :: Name of simulation directory.
 - get_faces :: If true, returns the coordinates of cell faces. Otherwise,
    returns the coordinates of cell centers.
Output:
 - z: Vertical level coordinates.
"""
function get_height(sim_dir::String; get_faces::Bool = false)
    z = nothing # Julia scoping
    try
        z = get_faces ? nc_fetch(sim_dir, "profiles", "zf") : nc_fetch(sim_dir, "profiles", "zc")
    catch e
        z = get_faces ? nc_fetch(sim_dir, "profiles", "z") : nc_fetch(sim_dir, "profiles", "z_half")
    end
    return z
end


"""
    normalize_profile(profile_vec, n_vars, var_vec)

Perform normalization of n_vars profiles contained in profile_vec
using the standard deviation associated with each variable, contained
in var_vec.
"""
function normalize_profile(profile_vec, n_vars, var_vec)
    prof_vec = deepcopy(profile_vec)
    dim_variable = Integer(length(profile_vec) / n_vars)
    for i in 1:n_vars
        prof_vec[(dim_variable * (i - 1) + 1):(dim_variable * i)] =
            prof_vec[(dim_variable * (i - 1) + 1):(dim_variable * i)] ./ sqrt(var_vec[i])
    end
    return prof_vec
end


"""
    get_time_covariance(sim_dir::String,
                     var_name::Array{String,1};
                     ti::Float64=0.0,
                     tf=0.0,
                     get_faces=false,
                     z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
                     normalize=false)

Obtain the covariance matrix of a group of profiles, where the covariance
is obtained in time.
Inputs:
 - sim_dir :: Name of simulation directory.
 - var_name :: List of variable names to be included.
 - ti, tf :: Initial and final times defining averaging interval.
 - z_scm :: If given, interpolates covariance matrix to this locations.
 - normalize :: Boolean specifying variable normalization.
"""
function get_time_covariance(
    m::ReferenceModel,
    sim_dir::String,
    var_names::Vector{String};
    get_faces = false,
    z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
)

    t = nc_fetch(sim_dir, "timeseries", "t")
    # Find closest interval in data
    ti_index = argmin(broadcast(abs, t .- m.t_start))
    tf_index = argmin(broadcast(abs, t .- m.t_end))
    ts_vec = zeros(0, length(ti_index:tf_index))
    num_outputs = length(var_names)
    pool_var = zeros(num_outputs)

    for i in 1:num_outputs
        var_ = nc_fetch(sim_dir, "profiles", var_names[i])
        # LES vertical fluxes are per volume, not mass
        if occursin("resolved_z_flux", var_names[i])
            rho_half = nc_fetch(sim_dir, "reference", "rho0_half")
            var_ = var_ .* rho_half
        end
        # Store pooled variance
        pool_var[i] = mean(var(var_[:, ti_index:tf_index], dims = 2))  # vertically averaged time-variance of variable
        # normalize timeseries
        ts_var_i = var_[:, ti_index:tf_index] ./ sqrt(pool_var[i])
        # Interpolate in space
        if !isnothing(z_scm)
            z_les = get_height(sim_dir, get_faces = get_faces)
            # Create interpolant
            ts_var_i_itp = interpolate((z_les, 1:(tf_index - ti_index + 1)), ts_var_i, (Gridded(Linear()), NoInterp()))
            # Interpolate
            ts_var_i = ts_var_i_itp(z_scm, 1:(tf_index - ti_index + 1))
        end
        ts_vec = cat(ts_vec, ts_var_i, dims = 1)  # dims: (Nz*num_outputs, Nt)
    end
    cov_mat = cov(ts_vec, dims = 2)  # covariance, w/ samples across time dimension (t_inds).
    return cov_mat, pool_var
end


function get_les_names(scm_y_names::Array{String, 1}, sim_dir::String)
    y_names = deepcopy(scm_y_names)
    if "thetal_mean" in y_names
        if occursin("GABLS", sim_dir) || occursin("Soares", sim_dir)
            y_names[findall(x -> x == "thetal_mean", y_names)] .= "theta_mean"
        else
            y_names[findall(x -> x == "thetal_mean", y_names)] .= "thetali_mean"
        end
    end
    if "total_flux_qt" in y_names
        y_names[findall(x -> x == "total_flux_qt", y_names)] .= "resolved_z_flux_qt"
    end
    if "total_flux_h" in y_names && (occursin("GABLS", sim_dir) || occursin("Soares", sim_dir))
        y_names[findall(x -> x == "total_flux_h", y_names)] .= "resolved_z_flux_theta"
    elseif "total_flux_h" in y_names
        y_names[findall(x -> x == "total_flux_h", y_names)] .= "resolved_z_flux_thetali"
    end
    if "u_mean" in y_names
        y_names[findall(x -> x == "u_mean", y_names)] .= "u_translational_mean"
    end
    if "v_mean" in y_names
        y_names[findall(x -> x == "v_mean", y_names)] .= "v_translational_mean"
    end
    if "tke_mean" in y_names
        y_names[findall(x -> x == "tke_mean", y_names)] .= "tke_nd_mean"
    end
    return y_names
end


function nc_fetch(dir, nc_group, var_name)
    find_prev_to_name(x) = occursin("Output", x)
    split_dir = split(dir, ".")
    sim_name = split_dir[findall(find_prev_to_name, split_dir)[1] + 1]
    ds = NCDataset(string(dir, "/stats/Stats.", sim_name, ".nc"))
    ds_group = ds.group[nc_group]
    ds_var = deepcopy(Array(ds_group[var_name]))
    close(ds)
    return Array(ds_var)
end


"""
    compute_errors(g_arr, y)

Computes the L2-norm error of each elmt of g_arr
wrt vector y.
"""
function compute_errors(g_arr, y)
    diffs = [g - y for g in g_arr]
    errors = map(x -> dot(x, x), diffs)
    return errors
end


"""
    penalize_nan(arr::Array{Float64, 1}; penalization::Float64 = 1.0e5)

Substitutes all NaN entries in `arr` by a penalization factor.
"""
function penalize_nan(arr::Array{Float64, 1}; penalization::Float64 = 1.0e5)
    return map(elem -> isnan(elem) ? penalization : elem, arr)
end


"""
    cov_from_cov_list(cov_list::Array{Array{FT,2},1}; indices=nothing)

Returns a block-diagonal covariance matrix constructed from covariances
within cov_list given by the indices. If isempty(indices), use all 
covariances to construct block-diagonal matrix.
"""
function cov_from_cov_list(cov_list::Array{Array{FT, 2}, 1}; indices = []) where {FT <: AbstractFloat}
    size_ = isempty(indices) ? sum([length(cov[1, :]) for cov in cov_list]) :
        sum([length(cov[1, :]) for (i, cov) in enumerate(cov_list) if i in indices])

    cov_ = zeros(size_, size_)
    vars_num = 1
    for (index, small_cov) in enumerate(cov_list)
        if index in indices
            vars = length(small_cov[1, :])
            cov_[vars_num:(vars_num + vars - 1), vars_num:(vars_num + vars - 1)] = small_cov
            vars_num = vars_num + vars
        end
    end
    return cov_
end

"""
    vec_from_vec_list(vec_list::Array{Array{FT,1},1}; indices=[], return_mapping=false)

Returns a vector constructed from vectors within vec_list given by the
indices. If isempty(indices), use all vectors to construct returned vector.
If return_mapping, function returns the positions of all the elements used
to construct the returned vector.
"""
function vec_from_vec_list(
    vec_list::Array{Array{FT, 1}, 1};
    indices = [],
    return_mapping = false,
) where {FT <: AbstractFloat}
    vector_ = zeros(0)
    elmt_num = []
    chosen_elmt_num = []
    for (index, small_vec) in enumerate(vec_list)
        index < 2 ? append!(elmt_num, 1:length(small_vec)) :
        append!(elmt_num, (elmt_num[end] + 1):(elmt_num[end] + length(small_vec)))
        if index in indices
            append!(vector_, small_vec)
            append!(chosen_elmt_num, (elmt_num[end] - length(small_vec) + 1):elmt_num[end])
        end
    end
    if return_mapping
        return vector_, chosen_elmt_num
    else
        return vector_
    end
end

"""
    serialize_struct(s::T) where {T}

Serializes the given structure as a dictionary to
allow storage in JLD2 format.
"""
function serialize_struct(s::T) where {T}
    keys = propertynames(s)
    vals = getproperty.(Ref(s), keys)
    return Dict(zip(string.(keys), vals))
end

"""
    deserialize_struct(dict::Dict{String}, ::Type{T})

Deserializes the given dictionary and constructs a struct
of the given type with the dictionary values.
"""
deserialize_struct(dict::Dict{String}, ::Type{T}) where {T} = T(map(fn -> dict["$fn"], fieldnames(T))...)

"""
    jld2_path(root::String, identifier::Union{String, Int}, prefix::String)

Generates a JLD2 path, given a root path, an identifier and a prefix.
"""
function jld2_path(root::String, identifier::Union{String, Int}, prefix::String)
    return joinpath(root, "$(prefix)$(identifier).jld2")
end

scm_init_path(root, version; prefix = "scm_initializer_") = jld2_path(root, version, prefix)
scm_output_path(root, version; prefix = "scm_output_") = jld2_path(root, version, prefix)
ekobj_path(root, iter; prefix = "ekobj_iter_") = jld2_path(root, iter, prefix)


"""
    write_versions(versions::Vector{Int}, iteration::Int; outdir_path::String = pwd())

Writes versions associated with an EnsembleKalmanProcess iteration to a text file.
"""
function write_versions(versions::Vector{Int}, iteration::Int; outdir_path::String = pwd())
    open(joinpath(outdir_path, "versions_$(iteration).txt"), "w") do io
        for version in versions
            write(io, "$(version)\n")
        end
    end
end
