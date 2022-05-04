"""
    HelperFuncs

Generic utils.
"""
module HelperFuncs

# We can work on qualifying these later
export vertical_interpolation,
    nc_fetch_interpolate,
    fetch_interpolate_transform,
    get_height,
    normalize_profile,
    nc_fetch,
    is_face_variable,
    get_stats_path,
    compute_mse,
    penalize_nan,
    serialize_struct,
    deserialize_struct,
    jld2_path,
    scm_init_path,
    scm_output_path,
    scm_val_init_path,
    scm_val_output_path,
    ekobj_path,
    write_versions,
    expand_dict_entry,
    get_entry,
    change_entry!

using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob
using JSON
using Random

"""
    vertical_interpolation(
        var_name::String,
        filename::String,
        z_scm::Vector{FT};
    ) where {FT <: AbstractFloat}

Returns the netcdf variable var_name interpolated to heights z_scm.

Inputs:
 - `var_name` :: Name of variable in the netcdf dataset.
 - `filename` :: nc filename
 - `z_scm` :: Vertical coordinate vector onto which var_name is interpolated.
Output:
 - The interpolated vector.
"""
function vertical_interpolation(var_name::String, filename::String, z_scm::Vector{FT};) where {FT <: AbstractFloat}
    z_ref = get_height(filename, get_faces = is_face_variable(filename, var_name))
    var_ = nc_fetch(filename, var_name)
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
        filename::String,
        z_scm::Union{Vector{<:Real}, Nothing};
    )

Returns the netcdf variable var_name, possibly interpolated to heights z_scm.

Inputs:
 - `var_name` :: Name of variable in the netcdf dataset.
 - `filename` :: nc filename
 - `z_scm` :: Vertical coordinate vector onto which var_name is interpolated.
Output:
 - The interpolated vector.
"""
function nc_fetch_interpolate(var_name::String, filename::String, z_scm::Union{Vector{<:Real}, Nothing};)
    if !isnothing(z_scm)
        return vertical_interpolation(var_name, filename, z_scm)
    else
        return nc_fetch(filename, var_name)
    end
end

"""
    fetch_interpolate_transform(
        var_name::String,
        filename::String,
        z_scm::Union{Vector{<:Real}, Nothing};
    )

Returns the netcdf variable var_name, possibly interpolated to heights z_scm. If the
variable needs to be transformed to be equivalent to an SCM variable, applies the
transformation as well.

Inputs:
 - `var_name` :: Name of variable in the netcdf dataset.
 - `filename` :: nc filename
 - `z_scm` :: Vertical coordinate vector onto which var_name is interpolated.
Output:
 - The interpolated and transformed vector.
"""
function fetch_interpolate_transform(var_name::String, filename::String, z_scm::Union{Vector{<:Real}, Nothing};)
    # PyCLES vertical fluxes are per volume, not mass
    if occursin("resolved_z_flux", var_name)
        var_ = nc_fetch_interpolate(var_name, filename, z_scm)
        rho_half = nc_fetch_interpolate("rho0_half", filename, z_scm)
        var_ = var_ .* rho_half
    elseif var_name == "horizontal_vel"
        u_ = nc_fetch_interpolate("u_mean", filename, z_scm)
        v_ = nc_fetch_interpolate("v_mean", filename, z_scm)
        var_ = sqrt.(u_ .^ 2 + v_ .^ 2)
    else
        var_ = nc_fetch_interpolate(var_name, filename, z_scm)
    end
    return var_
end

"""
    get_height(filename::String; get_faces::Bool = false)

Returns the vertical cell centers or faces of the given configuration.

Inputs:
 - filename :: nc filename.
 - get_faces :: If true, returns the coordinates of cell faces. Otherwise,
    returns the coordinates of cell centers.
Output:
 - z: Vertical level coordinates.
"""
function get_height(filename::String; get_faces::Bool = false)
    if get_faces
        return nc_fetch(filename, ("zf", "z"))
    else
        return nc_fetch(filename, ("zc", "z_half"))
    end
end

"""
    normalize_profile(profile_vec, n_vars, var_vec)

Perform normalization of n_vars profiles contained in profile_vec
using the variance associated with each variable, contained
in var_vec.
"""
function normalize_profile(profile_vec, n_vars, var_vec)
    y = deepcopy(profile_vec)
    var_dof = Integer(length(profile_vec) / n_vars)
    for i in 1:n_vars
        y[(var_dof * (i - 1) + 1):(var_dof * i)] = y[(var_dof * (i - 1) + 1):(var_dof * i)] ./ sqrt(var_vec[i])
    end
    return y
end

"""
    nc_fetch(filename::String, var_names::NTuple{N, Tuple}) where {N}
    nc_fetch(filename::String, var_name::String)

Returns the data for a variable `var_name` (or
tuple of strings, `varnames`), looping through
all dataset groups.
"""
function nc_fetch(filename::String, var_names::Tuple)
    NCDataset(filename) do ds
        for var_name in var_names
            if haskey(ds, var_name)
                return Array(ds[var_name])
            else
                for group_option in ["profiles", "reference", "timeseries"]
                    haskey(ds.group, group_option) || continue
                    if haskey(ds.group[group_option], var_name)
                        return Array(ds.group[group_option][var_name])
                    end
                end
            end
        end
        error("Variables $var_names not found in the output netCDF file $filename.")
    end
end
nc_fetch(filename::String, var_name::String) = nc_fetch(filename, (var_name,))

"""
    is_face_variable(filename::String, var_name::String)

A `Bool` indicating whether the given
variables is defined in faces, or not.
"""
function is_face_variable(filename::String, var_name::String)
    NCDataset(filename) do ds
        for group_option in ["profiles", "reference", "timeseries"]
            haskey(ds.group, group_option) || continue
            if haskey(ds.group[group_option], var_name)
                var_dims = dimnames(ds.group[group_option][var_name])
                if ("zc" in var_dims) | ("z_half" in var_dims)
                    return false
                elseif ("zf" in var_dims) | ("z" in var_dims)
                    return true
                else
                    error("Variable $var_name does not contain a vertical coordinate.")
                end
            end
        end
    end
end

"""
    get_stats_path(dir)

Given directory to standard LES or SCM output, fetch path to stats file.
"""
function get_stats_path(dir)
    stats = joinpath(dir, "stats")
    if !ispath(stats)
        stat_files = glob(relpath(abspath(joinpath(dir, "*.nc"))))
        @assert length(stat_files) == 1 "$(length(stat_files)) stats files found with paths $stat_files"
        return stat_files[1]
    end
    try
        stat_files = glob(relpath(abspath(joinpath(stats, "*.nc"))))
        @assert length(stat_files) == 1 "$(length(stat_files)) stats files found with paths $stat_files"
        return stat_files[1]
    catch e
        if isa(e, AssertionError)
            @warn "No unique stats netCDF file found in $stats. Extending search to other files."
            try
                stat_files = readdir(stats, join = true) # WindowsOS/julia relpath bug
                if length(stat_files) == 1
                    return stat_files[1]
                else
                    @error "No unique stats file found at $dir. The search returned $(length(stat_files)) results."
                end
            catch f
                if isa(f, Base.IOError)
                    @warn "Extended search errored with: $f"
                    return ""
                else
                    throw(f)
                end
            end
        else
            @warn "An error occurred retrieving the stats path at $dir. Throwing..."
            throw(e)
        end
    end
end

"""
    compute_mse(g_arr::Vector{Vector{FT}}, y::Vector{FT})::Vector{FT}
    compute_mse(g_mat::Matrix{FT}, y::Vector{FT})::Vector{FT}

Computes the L2-norm error of each vector, column or row of an array
with respect to a vector y.

Output:
 - The mse for each ensemble member.
"""
function compute_mse(g_arr::Vector{Vector{FT}}, y::Vector{FT})::Vector{FT} where {FT <: Real}
    diffs = [g - y for g in g_arr]
    errors = map(x -> dot(x, x) / length(x), diffs)
    return errors
end
function compute_mse(g_mat::Matrix{FT}, y::Vector{FT})::Vector{FT} where {FT <: Real}
    # Data are columns
    if size(g_mat, 1) == length(y)
        diffs = [g - y for g in eachcol(g_mat)]
        return map(x -> dot(x, x) / length(x), diffs)
        # Data are rows
    elseif size(g_mat, 2) == length(y)
        diffs = [g - y for g in eachrow(g_mat)]
        return map(x -> dot(x, x) / length(x), diffs)
    else
        throw(BoundsError("Dimension mismatch between inputs to `compute_error`."))
    end
end

"""
    penalize_nan(arr::Vector{FT}; penalization::FT = 1.0e5) where {FT <: AbstractFloat}

Substitutes all NaN entries in `arr` by a penalization factor.
"""
function penalize_nan(arr::Vector{FT}; penalization::FT = 1.0e5) where {FT <: AbstractFloat}
    return map(elem -> isnan(elem) ? penalization : elem, arr)
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
scm_val_init_path(root, version; prefix = "scm_val_initializer_") = jld2_path(root, version, prefix)
scm_val_output_path(root, version; prefix = "scm_val_output_") = jld2_path(root, version, prefix)
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

"""
    get_entry(dict, key, default)

Calls `get` but logs whether the default is used.
"""
function get_entry(dict, key, default)
    if haskey(dict, key)
        return dict[key]
    else
        @info "Key $key not found in dictionary. Returning default value $default."
        return default
    end
end

"""
    change_entry!(dict, keys_and_value)

Changes the entry of a nested dictionary, giving a tuple of all its keys and the new value

Inputs:
 - `dict`           :: Parent dictionary with an arbitrary number of nested dictionaries.
 - `keys_and_value` :: Tuple of keys from the parent dictionary to the entry to be modified,
                     and the value to use to modify it.
"""
function change_entry!(dict, keys_and_value)
    function get_last_nested_dict(dict, keys)
        if length(keys) > 1
            return get_last_nested_dict(dict[keys[1]], keys[2:end])
        else
            return dict
        end
    end
    # Unpack
    value = keys_and_value[end]
    keys = keys_and_value[1:(end - 1)]
    # Modify entry
    last_dict = get_last_nested_dict(dict, keys)
    last_dict[keys[end]] = value
end

end # module
