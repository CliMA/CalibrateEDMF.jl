#= Generic utils. =#

using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob
using JSON
using Random
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.LESUtils
using CalibrateEDMF.ModelTypes
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributionStorage
# TurbulenceConvection.jl
using TurbulenceConvection
tc_dir = dirname(dirname(pathof(TurbulenceConvection)));
include(joinpath(tc_dir, "integration_tests", "utils", "main.jl"))
include(joinpath(tc_dir, "integration_tests", "utils", "generate_namelist.jl"))

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
        println("Note: Δt_start > dt, which means that simulation stopped before reaching the requested t_start.")
        println("Requested t_start = $ti s. However, the last time available is $(t[end]) s.")
        println("Defaulting to penalized profiles...")
        for i in 1:length(var_names)
            var_ = get_height(sim_dir)
            append!(y, 1.0e5 * ones(length(var_[:])))
        end
        return y
    end
    if !isnothing(tf)
        Δt_end, tf_index = findmin(broadcast(abs, t .- tf))
        if Δt_end > dt
            println("Note: Δt_end > dt, which means that simulation stopped before reaching the requested t_end.")
            println("Requested t_end = $tf s. However, the last time available is $(t[end]) s.")
            println("Defaulting to penalized profiles...")
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
    vertical_interpolation(
        var_name::String,
        sim_dir::String,
        z_scm::Vector{FT};
        group::String = "profiles",
    ) where {FT <: AbstractFloat}

Returns the netcdf variable var_name interpolated to heights z_scm.

Inputs:
 - var_name :: Name of variable in the netcdf dataset.
 - sim_dir :: Name of simulation directory.
 - z_scm :: Vertical coordinate vector onto which var_name is interpolated.
 - group :: netcdf group of the variable.
Output:
 - The interpolated vector.
"""
function vertical_interpolation(
    var_name::String,
    sim_dir::String,
    z_scm::Vector{FT};
    group::String = "profiles",
) where {FT <: AbstractFloat}
    z_ref = get_height(sim_dir, get_faces = is_face_variable(sim_dir, group, var_name))
    var_ = nc_fetch(sim_dir, group, var_name)
    if length(size(var_)) == 2
        # Create interpolant
        nodes = (z_ref, 1:size(var_, 2))
        var_itp = extrapolate(interpolate(nodes, var_, (Gridded(Linear()), NoInterp())), Line())
        # Return interpolated vector
        return var_itp(z_scm, 1:size(var_, 2))
    elseif length(size(var_)) == 1
        # Create interpolant
        nodes = (z_ref,)
        var_itp = LinearInterpolation(nodes, var_; extrapolation_bc = Line())
        # Return interpolated vector
        return var_itp(z_scm)
    end
end

"""
    nc_fetch_interpolate(
        var_name::String,
        sim_dir::String,
        z_scm::Union{Vector{Float64}, Nothing};
        group::String = "profiles",
    )

Returns the netcdf variable var_name, possibly interpolated to heights z_scm.

Inputs:
 - var_name :: Name of variable in the netcdf dataset.
 - sim_dir :: Name of simulation directory.
 - z_scm :: Vertical coordinate vector onto which var_name is interpolated.
 - group :: netcdf group of the variable.
Output:
 - The interpolated vector.
"""
function nc_fetch_interpolate(
    var_name::String,
    sim_dir::String,
    z_scm::Union{Vector{Float64}, Nothing};
    group::String = "profiles",
)
    if !isnothing(z_scm)
        return vertical_interpolation(var_name, sim_dir, z_scm, group = group)
    else
        return nc_fetch(sim_dir, group, var_name)
    end
end

"""
    fetch_interpolate_transform(
        var_name::String,
        sim_dir::String,
        z_scm::Union{Vector{Float64}, Nothing};
        group::String = "profiles",
    )

Returns the netcdf variable var_name, possibly interpolated to heights z_scm. If the
variable needs to be transformed to be equivalent to an SCM variable, applies the
transformation as well.

Inputs:
 - var_name :: Name of variable in the netcdf dataset.
 - sim_dir :: Name of simulation directory.
 - z_scm :: Vertical coordinate vector onto which var_name is interpolated.
 - group :: netcdf group of the variable.
Output:
 - The interpolated and transformed vector.
"""
function fetch_interpolate_transform(
    var_name::String,
    sim_dir::String,
    z_scm::Union{Vector{Float64}, Nothing};
    group::String = "profiles",
)
    # PyCLES vertical fluxes are per volume, not mass
    if occursin("resolved_z_flux", var_name)
        var_ = nc_fetch_interpolate(var_name, sim_dir, z_scm, group = group)
        rho_half = nc_fetch_interpolate("rho0_half", sim_dir, z_scm, group = "reference")
        var_ = var_ .* rho_half
    elseif occursin("horizontal_vel", var_name)
        u_ = nc_fetch_interpolate("u_mean", sim_dir, z_scm, group = group)
        v_ = nc_fetch_interpolate("v_mean", sim_dir, z_scm, group = group)
        var_ = sqrt.(u_ .^ 2 + v_ .^ 2)
    else
        var_ = nc_fetch_interpolate(var_name, sim_dir, z_scm, group = group)
    end
    return var_
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
using the variance associated with each variable, contained
in var_vec.
"""
function normalize_profile(profile_vec, n_vars, var_vec)
    y = deepcopy(profile_vec)
    dim_variable = Integer(length(profile_vec) / n_vars)
    for i in 1:n_vars
        y[(dim_variable * (i - 1) + 1):(dim_variable * i)] =
            y[(dim_variable * (i - 1) + 1):(dim_variable * i)] ./ sqrt(var_vec[i])
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


function nc_fetch(dir, nc_group, var_name)
    ds = NCDataset(get_stats_path(dir))
    ds_group = ds.group[nc_group]
    ds_var = deepcopy(Array(ds_group[var_name]))
    close(ds)
    return Array(ds_var)
end

"""Returns whether the given variables is defined in faces, or not."""
function is_face_variable(dir, nc_group, var_name)
    ds = NCDataset(get_stats_path(dir))
    ds_group = ds.group[nc_group]
    var_dims = dimnames(ds_group[var_name])
    close(ds)
    if ("zc" in var_dims) | ("z_half" in var_dims)
        return false
    elseif ("zf" in var_dims) | ("z" in var_dims)
        return true
    else
        println("Variable $var_name does not contain a vertical dimension.")
        return false
    end
end

"""
    get_stats_path(dir)

Given directory to standard LES or SCM output, fetch path to stats file.
"""
function get_stats_path(dir)
    stats = joinpath(dir, "stats")
    try
        stat_files = glob(relpath(joinpath(stats, "*.nc")))
        @assert length(stat_files) == 1
        return stat_files[1]
    catch e
        if isa(e, AssertionError)
            @warn "No unique stats netCDF file found at $dir. Extending search to other files."
            stat_files = readdir(stats, join = true) # WindowsOS/julia 1.6.0 relpath bug
            if length(stat_files) == 1
                return stat_files[1]
            else
                @error "No unique stats file found at $dir."
            end
        else
            @error "An error occurred retrieving the stats path at $dir."
        end
    end
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
function penalize_nan(arr::Array{FT, 1}; penalization::FT = 1.0e5) where {FT <: AbstractFloat}
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
    vec_from_vec_list(vec_list::Vector{Vector{FT}}; indices=[], return_mapping=false)

Returns a vector constructed from vectors within vec_list given by the
indices. If isempty(indices), use all vectors to construct returned vector.
If return_mapping, function returns the positions of all the elements used
to construct the returned vector.
"""
function vec_from_vec_list(
    vec_list::Vector{Vector{FT}};
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

"Returns the N-vector stored in `dict[key]`, or an N-vector of `nothing`"
function expand_dict_entry(dict, key, N)
    val = get(dict, key, nothing)
    r = isnothing(val) ? repeat([nothing], N) : val
    @assert length(r) == N
    r
end
