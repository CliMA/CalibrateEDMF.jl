module LESUtils

import CalibrateEDMF.ReferenceModels: ReferenceModel
using TurbulenceConvection
tc = dirname(pathof(TurbulenceConvection))
include(joinpath(tc, "name_aliases.jl"))
include("helper_funcs.jl")

export get_les_names, get_cfsite_les_dir, find_alias


"""
    LES_library
Enumerate available LES simulations described in `Shen et al. 2021`.
The following cfsites are available across listed models, months,
and experiments. Although some additional simulations are available.
"""
LES_library = Dict(
    "cfsite_numbers" => collect(2:23),
    "forcing_models" => ["HadGEM2-A", "CNRM-CM5", "CNRM-CM6-1"],
    "months" => [1, 4, 7, 10],
    "experiments" => ["amip", "amip4K"],
)

"""
    get_les_names(y_names::Vector{String}, les_dir::String)
    get_les_names(m::ReferenceModel, les_dir::String)

Returns the aliases of the variables actually present in `les_dir`
corresponding to SCM variables `y_names`.
"""
get_les_names(m::ReferenceModel, les_dir::String)::Vector{String} = get_les_names(m.y_names, les_dir)
function get_les_names(y_names::Vector{String}, les_dir::String)::Vector{String}
    dict = name_aliases()
    y_alias_groups = [haskey(dict, var) ? (dict[var]..., var) : (var,) for var in y_names]
    return [find_alias(aliases, les_dir) for aliases in y_alias_groups]
end

"""
    find_alias(les_dir::String, aliases::Tuple{Vararg{String}})

Finds the alias present in an NCDataset from a list of possible aliases.
"""
function find_alias(aliases::Tuple{Vararg{String}}, les_dir::String)
    ds = NCDataset(get_stats_path(les_dir))
    for alias in aliases
        if haskey(ds, alias)
            return alias
        else
            for group_option in ["profiles", "reference", "timeseries"]
                haskey(ds.group, group_option) || continue
                if haskey(ds.group[group_option], alias)
                    return alias
                end
            end
        end
    end
    error("None of the aliases $aliases found in the dataset $les_dir.")
end

"""
    get_cfsite_les_dir(
        cfsite_number::Integer;
        forcing_model::String = "HadGEM2-A",
        month::Integer = 7,
        experiment::String = "amip",)

Given information about an LES run from [Shen2021](@cite),
fetch LES directory on central cluster.

    Inputs:
    - cfsite_number  :: cfsite number
    - forcing_model :: {"HadGEM2-A", "CNRM-CM5", "CNRM-CM6-1", "IPSL-CM6A-LR"} - name of climate model used for forcing
    - month :: {1, 4, 7, 10} - month of simulation
    - experiment :: {"amip", "amip4K"} - experiment from which LES was forced

   Outputs:
    - les_dir - path to les simulation containing stats folder

"""
function get_cfsite_les_dir(
    cfsite_number::Integer;
    forcing_model::String = "HadGEM2-A",
    month::Integer = 7,
    experiment::String = "amip",
)
    try
        @assert cfsite_number in LES_library["cfsite_numbers"]
        @assert forcing_model in LES_library["forcing_models"]
        @assert month in LES_library["months"]
        @assert experiment in LES_library["experiments"]
    catch
        throw(AssertionError("The requested cfsite LES does not exist."))
    end
    cfsite_number = string(cfsite_number)
    month = string(month, pad = 2)
    root_dir = "/central/groups/esm/zhaoyi/GCMForcedLES/cfsite/$month/$forcing_model/$experiment/"
    rel_dir = join(["Output.cfsite$cfsite_number", forcing_model, experiment, "2004-2008.07.4x"], "_")
    return joinpath(root_dir, rel_dir)
end

end # module
