#=

Helper functions for SOCRATES data processing

=#

package_dir = dirname(@__DIR__) # parent of parent...

function get_socrates_dir(
    flight_number::Integer,
    forcing_tupe::String;
    )

    case_name = "RF"*string(flight_number,pad=2)*"_"*forcing_type # folder name where were stored the data
    return joinpath(package_dir , "experiments" , "SOCRATES_Test_Dynamical_Calibration" , "Truth" , case_name , "stats.nc" ) # the total path, stats.nc is the name expected here in CEDMF.jl






end

# I'm not sure if we need these for socrates but it's also from TurbulenceConvection.jl so maybe we do? But the names won't all match the atlas results so maybe this doesnt work? I'm not sure if this is strictly for the atlas stuff or also for the new runs we're doing in TC here...
"""
get_socrates_names(y_names::Vector{String}, filename::String)
get_socrates_names(m::ReferenceModel, filename::String)

Returns the aliases of the variables actually present in the nc
file (`filename`) corresponding to SCM variables `y_names`.
"""
get_socrates_names(m::ReferenceModel, filename::String)::Vector{String} = get_les_names(m.y_names, filename)
function get_socrates_names(y_names::Vector{String}, filename::String)::Vector{String}
    dict = name_aliases()
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
