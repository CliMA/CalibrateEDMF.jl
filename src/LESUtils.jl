module LESUtils

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

"""
    get_cfsite_les_dir(
        cfsite_number::Integer;
        forcing_model::String = "HadGEM2-A",
        month::Integer = 7,
        experiment::String = "amip",)

Given information about an LES run from `Shen et al. 2021`,
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
        println("The requested cfsite LES does not exist.")
    end
    cfsite_number = string(cfsite_number)
    month = string(month, pad = 2)
    root_dir = "/central/groups/esm/zhaoyi/GCMForcedLES/cfsite/$month/$forcing_model/$experiment/"
    rel_dir = join(["Output.cfsite$cfsite_number", forcing_model, experiment, "2004-2008.07.4x"], "_")
    return joinpath(root_dir, rel_dir)
end

end # module
