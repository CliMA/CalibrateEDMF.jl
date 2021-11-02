module LESUtils
import CalibrateEDMF.ReferenceModels: ReferenceModel

export get_les_names, get_cfsite_les_dir


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

get_les_names(m::ReferenceModel)::Vector{String} = get_les_names(m.y_names, m.case_name)
function get_les_names(scm_y_names::Vector{String}, case_name::String)::Vector{String}
    y_names = deepcopy(scm_y_names)
    if (case_name == "GABLS") || (case_name == "Soares")
        y_names[y_names .== "thetal_mean"] .= "theta_mean"
        y_names[y_names .== "total_flux_h"] .= "resolved_z_flux_theta"
    else
        y_names[y_names .== "thetal_mean"] .= "thetali_mean"
        y_names[y_names .== "total_flux_h"] .= "resolved_z_flux_thetali"
    end
    y_names[y_names .== "total_flux_qt"] .= "resolved_z_flux_qt"
    y_names[y_names .== "u_mean"] .= "u_translational_mean"
    y_names[y_names .== "v_mean"] .= "v_translational_mean"
    y_names[y_names .== "tke_mean"] .= "tke_nd_mean"
    return y_names
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
        @error "The requested cfsite LES does not exist."
    end
    cfsite_number = string(cfsite_number)
    month = string(month, pad = 2)
    root_dir = "/central/groups/esm/zhaoyi/GCMForcedLES/cfsite/$month/$forcing_model/$experiment/"
    rel_dir = join(["Output.cfsite$cfsite_number", forcing_model, experiment, "2004-2008.07.4x"], "_")
    return joinpath(root_dir, rel_dir)
end

end # module
