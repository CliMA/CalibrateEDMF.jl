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
    is_timeseries,
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
    change_entry!,
    ParameterMap,
    do_nothing_param_map,
    expand_params,
    namelist_subdict_by_key

using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob
using JSON
using Random

"""
    struct ParameterMap

`ParameterMap` is a struct that defines relations between different parameters.

The dictionary `mapping` specifies a map from any parameter that is not being 
calibrated to either a calibrated parameter, or a number. 
If another parameter name is specified, the samples from that parameter's distribution
is used as the given parameter's value throughout calibration, effectively defining
an equivalence between the two parameters.
If a number is specified, the parameter will be assigned that constant value throughout 
the calibration process.

These mappings are primarily useful for vector parameters, where we can choose to only
calibrate specific components of a vector parameter, or setting different components of
a vector equal to other components of the vector (or to other parameters). Note, vector 
components are specified by `<name>_{<index>}`, e.g. `general_stochastic_ent_params_{3}` 
for the third component of the vector parameter `general_stochastic_ent_params`.

# Examples
Suppose we have the parameter vector `param` with five components. We can choose to only
calibrate the first and third component, fixing the fourth component to 3.0,
and also specifying that the first and second component should be equal, in the following way:
```jldoctest
julia> param_map = ParameterMap(
    mapping = Dict(
        "param_{2}" => "param_{1}",
        "param_{4}" => 3.0,
    ),
)
```

The parameter map should be specified in the function `get_prior_config()` in `config.jl`,
```jldoctest
# function get_prior_config()  
config = Dict()

config["constraints"] = Dict(
    "param_{1}" => [bounded(1.0, 2.0)],
    "param_{3} => [bounded(3.0, 4.0)],
)

config["param_map"] = CalibrateEDMF.HelperFuncs.ParameterMap(
    mapping = Dict(
        "param_{2}" => "param_{1}",
        "param_{4}" => 3.0,
    ),
)

# ...
# end
```
Notice that the fifth component was neither specified in the prior, nor the mapping. This will fix
that component to its default value as specified in the namelist (i.e. in `TurbulenceConvection.jl`).

"""
Base.@kwdef struct ParameterMap
    mapping::Dict
end

""" do-nothing param_map."""
do_nothing_param_map() = ParameterMap(Dict())

"""
    expand_params(u_names_calib, u_calib, param_map, namelist)

Expand a list of parameters using a param_map and a namelist.

Expand a list of parameters and its corresponding values using the [`ParameterMap`](@ref). 
If `u_names_calib` contain vector components, fetch all unspecified components 
from the `namelist` so that for any vector component in `u_names_calib`, 
all other components of that vector are also specified.

# Arguments
- `u_names_calib`: A list of parameter names that are being calibrated
- `u_calib`: A list of values associated with the parameter names
- `param_map`: A [`ParameterMap`](@ref) specifying a parameter mapping.
- `namelist`: A dictionary of default parameter values.

Returns a tuple of two vectors defining parameter names and parameter values,
possibly expanded relative to the input arguments to define all components of
vector parameters or from relations specified by the [`ParameterMap`](@ref).
"""
function expand_params(u_names_calib::AbstractVector, u_calib::AbstractVector, param_map::ParameterMap, namelist::Dict)

    # Check that `param_map` and `u_names_calib` do not contain any of the same parameters
    mapping_keys = collect(keys(param_map.mapping))
    if any(mapping_keys .∈ (u_names_calib,))
        illegal_params = join(mapping_keys[mapping_keys .∈ (u_names_calib,)], ", ")
        throw(
            ArgumentError(
                "Parameters cannot simultaneously be calibrated and fixed with a param_map. \n Check config for: $illegal_params",
            ),
        )
    end

    # fill `params` using `namelist`
    base_params = Dict(u_names_calib .=> u_calib)
    params = fill_param_vectors(namelist, base_params)
    # update `params` using `param_map` 
    for (mapping_keys, mapping_values) in param_map.mapping
        params[mapping_keys] = if isa(mapping_values, Number)
            mapping_values  # Set numeric value if explicitly provided
        elseif isa(mapping_values, AbstractString)
            # if `p1` => `p2`, get value associated with `p2`
            @assert haskey(base_params, mapping_values) "Parameter `$mapping_values` not among calibrated parameters! Check your config."
            base_params[mapping_values]
        else
            throw(
                ArgumentError(
                    "The type of `$mapping_values` for the Pair `$mapping_keys => $mapping_values` is not recognized. " *
                    "Should be `String` or `Number`, but has type `$(typeof(mapping_values))`.",
                ),
            )
        end
    end

    return collect.((keys(params), values(params)))
end

"""
    fill_param_vectors(namelist, params)

Expand `params` dictionary with defaults from `namelist`.

In practice, `params` is only expanded if it contains vector components,
in which case default values of unspecified components are fetched from
the namelist and appended to `params`.
"""
function fill_param_vectors(namelist::Dict, params::Dict{<:AbstractString, <:Number})
    # Fetch default vector parameters from `namelist` that match `params`
    vec_param_names = unique(getfield.(filter(!isnothing, match.(r".*(?=_{\d+})", keys(params))), :match))
    namelist_vec_params = Dict(vec_param_names .=> get_namelist_value.(Ref(namelist), vec_param_names))
    flatten_vector_parameter(p) = "$(p.first)_{" .* string.(1:length(p.second)) .* "}" .=> p.second
    vec_params = Dict((map(flatten_vector_parameter, collect(namelist_vec_params))...)...)
    # Update parameter list
    return merge(vec_params, params)
end

"""
    namelist_subdict_by_key(namelist, param_name) -> subdict

Return the subdict of `namelist` that has `param_name` as a key.

The namelist is defined in TurbulenceConvection.jl
"""
function namelist_subdict_by_key(namelist::Dict, param_name::AbstractString)::Dict
    subdict = if haskey(namelist["turbulence"]["EDMF_PrognosticTKE"], param_name)
        namelist["turbulence"]["EDMF_PrognosticTKE"]
    elseif haskey(namelist["microphysics"], param_name)
        namelist["microphysics"]
    elseif haskey(namelist["time_stepping"], param_name)
        namelist["time_stepping"]
    else
        throw(ArgumentError("Parameter $param_name cannot be calibrated. Consider adding namelist dictionary if needed."))
    end
    return subdict
end

""" Get the namelist value for some parameter name"""
get_namelist_value(namelist::Dict, param_name::AbstractString) =
    namelist_subdict_by_key(namelist, param_name)[param_name]

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
    if ndims(var_) == 2
        # Create interpolant
        nodes = (z_ref, 1:size(var_, 2))
        var_itp = extrapolate(interpolate(nodes, var_, (Gridded(Linear()), NoInterp())), Line())
        # Return interpolated vector
        return var_itp(z_scm, 1:size(var_, 2))
    elseif ndims(var_) == 1
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
    if !is_timeseries(filename, var_name) && !isnothing(z_scm)
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
    normalize_profile(
        y::Array{FT},
        norm_vec::Array{FT},
        prof_dof::IT,
        prof_indices::Union{BitVector, Nothing} = nothing,
    ) where {FT <: Real, IT <: Integer}

Perform normalization of the aggregate observation vector `y` using separate
normalization constants for each variable, contained in `norm_vec`.

Inputs:
 - `y` :: Aggregate observation vector.
 - `norm_vec` :: Vector of squares of normalization factors.
 - `prof_dof` :: Degrees of freedom of vertical profiles contained in `y`.
 - `prof_indices` :: Vector of booleans specifying which variables are profiles, and which
    are timeseries.
Output:
 - The normalized aggregate observation vector.
"""
function normalize_profile(
    y::Array{FT},
    norm_vec::Array{FT},
    prof_dof::IT,
    prof_indices::Union{Vector{Bool}, Nothing} = nothing,
) where {FT <: Real, IT <: Integer}
    y_ = deepcopy(y)
    n_vars = length(norm_vec)
    prof_indices = isnothing(prof_indices) ? repeat([true], n_vars) : prof_indices
    loc_start = 1
    for i in 1:n_vars
        loc_end = prof_indices[i] ? loc_start + prof_dof - 1 : loc_start
        y_[loc_start:loc_end] = y_[loc_start:loc_end] ./ sqrt(norm_vec[i])
        loc_start = loc_end + 1
    end
    return y_
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

A `Bool` indicating whether the given variable is defined in a face,
or not (cell center).

TurbulenceConvection data are consistent, meaning that variables at
cell faces (resp. centers) have as dim `zf` (resp., `zc`).

PyCLES variables are inconsistent. All variables have as a dim `z`,
the cell face locations, but most of them (except for the statistical
moments of w) are actually defined at cell centers (`z_half`). 
"""
function is_face_variable(filename::String, var_name::String)
    # PyCLES cell face variables
    pycles_face_vars = ["w_mean", "w_mean2", "w_mean3"]
    NCDataset(filename) do ds
        for group_option in ["profiles", "reference"]
            haskey(ds.group, group_option) || continue
            if haskey(ds.group[group_option], var_name)
                var_dims = dimnames(ds.group[group_option][var_name])
                if ("zc" in var_dims) | ("z_half" in var_dims)
                    return false
                elseif ("zf" in var_dims) | (var_name in pycles_face_vars)
                    return true
                elseif ("z" in var_dims) # "Inconsistent" PyCLES variables
                    return false
                else
                    throw(ArgumentError("Variable $var_name does not contain a vertical coordinate."))
                end
            end
        end
    end
end

"""
    is_timeseries(filename::String, var_name::String)

A `Bool` indicating whether the given variable is a timeseries.
"""
function is_timeseries(filename::String, var_name::String)
    NCDataset(filename) do ds
        if haskey(ds.group, "timeseries")
            if haskey(ds.group["timeseries"], var_name)
                return true
            else
                return false
            end
        else
            return false
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
        nc_path = joinpath(dir, "*.nc")
        stat_files = glob(relpath(abspath(nc_path)))
        @assert length(stat_files) == 1 "$(length(stat_files)) stats files found with paths $nc_path"
        return stat_files[1]
    end
    try
        nc_path = joinpath(stats, "*.nc")
        stat_files = glob(relpath(abspath(nc_path)))
        @assert length(stat_files) == 1 "$(length(stat_files)) stats files found with paths $nc_path"
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
    write_versions(versions, iteration; outdir_path = pwd())

Writes versions associated with an EnsembleKalmanProcess iteration to a text file.
"""
function write_versions(versions::Vector{String}, iteration::Int; outdir_path::String = pwd())
    open(joinpath(outdir_path, "versions_$(iteration).txt"), "w") do io
        println.(Ref(io), versions)
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

Changes the entry of a nested dictionary, given a tuple of all its keys and the new value

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
