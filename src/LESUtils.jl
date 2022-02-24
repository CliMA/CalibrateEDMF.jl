module LESUtils

import NCDatasets
import ..ReferenceModels: ReferenceModel
import TurbulenceConvection
const TC = TurbulenceConvection
import ArtifactWrappers
const AW = ArtifactWrappers
using ..HelperFuncs

export get_les_names, get_cfsite_les_dir, find_alias, get_path_to_artifact


"""
    get_LES_library

Hierarchical dictionary of available LES simulations described in `Shen et al. 2021`.
The following cfsites are available across listed models, months,
and experiments.
"""
function get_LES_library()
    LES_library = Dict("HadGEM2-A" => Dict(), "CNRM-CM5" => Dict(), "CNRM-CM6-1" => Dict())
    Shen_et_al_sites = collect(2:15)
    append!(Shen_et_al_sites, collect(17:23))

    LES_library["HadGEM2-A"]["10"] = Dict()
    LES_library["HadGEM2-A"]["10"]["cfsite_numbers"] = Shen_et_al_sites
    LES_library["HadGEM2-A"]["07"] = Dict()
    LES_library["HadGEM2-A"]["07"]["cfsite_numbers"] = Shen_et_al_sites
    LES_library["HadGEM2-A"]["04"] = Dict()
    LES_library["HadGEM2-A"]["04"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [15, 17, 18])
    LES_library["HadGEM2-A"]["01"] = Dict()
    LES_library["HadGEM2-A"]["01"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [15, 17, 18, 19, 20])

    for month in ["01", "04", "07", "10"]
        LES_library["HadGEM2-A"][month]["experiments"] = ["amip", "amip4K"]
    end
    return LES_library
end

"""
    get_les_names(y_names::Vector{String}, les_dir::String)
    get_les_names(m::ReferenceModel, les_dir::String)

Returns the aliases of the variables actually present in `les_dir`
corresponding to SCM variables `y_names`.
"""
get_les_names(m::ReferenceModel, les_dir::String)::Vector{String} = get_les_names(m.y_names, les_dir)
function get_les_names(y_names::Vector{String}, les_dir::String)::Vector{String}
    dict = TC.name_aliases()
    y_alias_groups = [haskey(dict, var) ? (dict[var]..., var) : (var,) for var in y_names]
    return [find_alias(aliases, les_dir) for aliases in y_alias_groups]
end

"""
    find_alias(les_dir::String, aliases::Tuple{Vararg{String}})

Finds the alias present in an NCDataset from a list of possible aliases.
"""
function find_alias(aliases::Tuple{Vararg{String}}, les_dir::String)
    NCDatasets.NCDataset(get_stats_path(les_dir)) do ds
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
    - forcing_model :: {"HadGEM2-A", "CNRM-CM5", "CNRM-CM6-1", "IPSL-CM6A-LR"} - name of climate model used for forcing.
        Currently, only "HadGEM2-A" simulations are available reliably.
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

function get_path_to_artifact(casename = "Bomex", artifact_type = "PyCLES_output", artifact_dir = @__DIR__)
    local_to_box = Dict("Bomex" => "https://caltech.box.com/shared/static/d6oo7th33839qmp4z99n8z4ryk3iepoq.nc")
    if haskey(local_to_box, casename)
        #! format: off
        output_artifact = AW.ArtifactWrapper(
            artifact_dir,
            isempty(get(ENV, "CI", "")),
            artifact_type,
            AW.ArtifactFile[
            AW.ArtifactFile(url = local_to_box[casename], filename = string(casename, ".nc"),),
            ],
        )
        output_artifact_path = AW.get_data_folder(output_artifact)
        return output_artifact_path
    else
        throw(KeyError("Artifacts for casename $casename of type $artifact_type are not currently available."))
    end
end

end # module
