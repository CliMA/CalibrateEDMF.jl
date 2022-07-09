module LESUtils

export get_les_names, get_cfsite_les_dir, find_alias, get_path_to_artifact

import NCDatasets
import ..ReferenceModels: ReferenceModel
import TurbulenceConvection
const TC = TurbulenceConvection
using ..HelperFuncs


"""
    get_LES_library

Hierarchical dictionary of available LES simulations described in [Shen2022](@cite).
The following cfsites are available across listed models, months,
and experiments.
"""
function get_LES_library()
    LES_library = Dict("HadGEM2-A" => Dict(), "CNRM-CM5" => Dict(), "CNRM-CM6-1" => Dict())
    Shen_et_al_sites = collect(2:15)
    append!(Shen_et_al_sites, collect(17:23))
    Shen_et_al_deep_convection_sites = (collect(30:33)..., collect(66:70)..., 82, 92, 94, 96, 99, 100)
    append!(Shen_et_al_sites, Shen_et_al_deep_convection_sites)

    LES_library["HadGEM2-A"]["10"] = Dict()
    LES_library["HadGEM2-A"]["10"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [94, 100])
    LES_library["HadGEM2-A"]["07"] = Dict()
    LES_library["HadGEM2-A"]["07"]["cfsite_numbers"] = Shen_et_al_sites
    LES_library["HadGEM2-A"]["04"] = Dict()
    LES_library["HadGEM2-A"]["04"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [15, 17, 18, 32, 92, 94])
    LES_library["HadGEM2-A"]["01"] = Dict()
    LES_library["HadGEM2-A"]["01"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [15, 17, 18, 19, 20])

    for month in ["01", "04", "07", "10"]
        LES_library["HadGEM2-A"][month]["experiments"] = ["amip", "amip4K"]
    end
    return LES_library
end

"""
    get_les_names(y_names::Vector{String}, filename::String)
    get_les_names(m::ReferenceModel, filename::String)

Returns the aliases of the variables actually present in the nc
file (`filename`) corresponding to SCM variables `y_names`.
"""
get_les_names(m::ReferenceModel, filename::String)::Vector{String} = get_les_names(m.y_names, filename)
function get_les_names(y_names::Vector{String}, filename::String)::Vector{String}
    dict = TC.name_aliases()
    y_alias_groups = [haskey(dict, var) ? (dict[var]..., var) : (var,) for var in y_names]
    return [find_alias(aliases, filename) for aliases in y_alias_groups]
end

"""
    find_alias(aliases::Tuple{Vararg{String}}, filename::String)

Finds the alias present in an NCDataset from a list of possible aliases.
"""
function find_alias(aliases::Tuple{Vararg{String}}, filename::String)
    NCDatasets.NCDataset(filename) do ds
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
        error("None of the aliases $aliases found in the dataset $filename.")
    end
end

"""
    get_cfsite_les_dir(
        cfsite_number::Integer;
        forcing_model::String = "HadGEM2-A",
        month::Integer = 7,
        experiment::String = "amip",)

Given information about an LES run from [Shen2022](@cite),
fetch LES directory on central cluster.

Inputs:

 - cfsite_number  :: cfsite number
 - forcing_model :: {"HadGEM2-A", "CNRM-CM5", "CNRM-CM6-1", "IPSL-CM6A-LR"} - name of climate model used for forcing. Currently, only "HadGEM2-A" simulations are available reliably.
 - month :: {1, 4, 7, 10} - month of simulation.
 - experiment :: {"amip", "amip4K"} - experiment from which LES was forced.

Outputs:

 - les_dir - path to les simulation containing stats folder
"""
function get_cfsite_les_dir(
    cfsite_number::Integer;
    forcing_model::String = "HadGEM2-A",
    month::Integer = 7,
    experiment::String = "amip",
)
    month = string(month, pad = 2)
    try
        LES_library = get_LES_library()
        @assert forcing_model in keys(LES_library)
        @assert String(month) in keys(LES_library[forcing_model])
        @assert cfsite_number in LES_library[forcing_model][month]["cfsite_numbers"]
        @assert experiment in LES_library[forcing_model][month]["experiments"]
    catch e
        @error "The requested cfsite LES does not exist."
        throw(e)
    end
    cfsite_number = string(cfsite_number)
    root_dir = "/central/groups/esm/zhaoyi/GCMForcedLES/cfsite/$month/$forcing_model/$experiment/"
    rel_dir = join(["Output.cfsite$cfsite_number", forcing_model, experiment, "2004-2008.$month.4x"], "_")
    return joinpath(root_dir, rel_dir)
end

include("artifact_funcs.jl")

end # module
