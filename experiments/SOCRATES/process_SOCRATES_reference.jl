using CalibrateEDMF
using Dates
const NC = CalibrateEDMF.NetCDFIO.NC # is already loaded so let's not reload
flight_numbers = [1, 9, 10, 11, 12, 13]
forcing_types = [:obs_data, :ERA5_data]

# this_dir = @__DIR__ # the location of this file
this_dir = "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/"
# data_dir = joinpath(this_dir, "Reference", "Atlas_LES") # the folder where we store our truth (Atlas LES Data)

# The variables we need for calibration routine from the Atlas LES outputs
default_data_vars_atlas = Dict{Tuple{String, String}, Union{String, Nothing}}( #TC.jl (Name, Group) =>  Atlas Name
    ("thetal_mean", "profiles") => "THETAL",
    ("temperature_mean", "profiles") => "TABS",
    ("qt_mean", "profiles") => "QT",
    ("ql_mean", "profiles") => "QCL", # Seems QCL and QC are the same
    ("qi_mean", "profiles") => "QCI", # seems QCI and QI are the same
    ("qc_mean", "profiles") => "QN", # cloud liquid and ice (not in TC.jl output)
    ("qr_mean", "profiles") => "QR",
    ("qs_mean", "profiles") => "QS",
    ("qg_mean", "profiles") => "QG",
    ("qp_mean", "profiles") => "QP", # total precipitation (rain + snow) , not in TC.jl output
    ("t", "timeseries") => "time",
    ("t", "profiles") => "time",
    ("zf", "profiles") => "z", # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
    ("zf", "reference") => "z", # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
)


neg = x -> -x
null_func = x -> x
g_to_kg = x -> x ./ FT(1000)
perday_to_persec = x -> x ./ FT(24 * 3600)
w_to_q = w -> w ./ (1 .+ w) # mixing ratio to specific humidity    


default_var_LES_to_TC_scaling = Dict{Tuple{String, String}, Function}(
    ("thetal_mean", "profiles") => null_func,
    ("temperature_mean", "profiles") => null_func,
    ("qt_mean", "profiles") => w_to_q ∘ g_to_kg,
    ("ql_mean", "profiles") => w_to_q ∘ g_to_kg, # Seems QCL and QC are the same
    ("qi_mean", "profiles") => w_to_q ∘ g_to_kg, # seems QCI and QI are the same
    ("qr_mean", "profiles") => w_to_q ∘ g_to_kg,
    ("qs_mean", "profiles") => w_to_q ∘ g_to_kg,
    #
    ("qc_mean", "profiles") => w_to_q ∘ g_to_kg, # cloud liquid and ice (not in TC.jl output)
    ("qg_mean", "profiles") => w_to_q ∘ g_to_kg,
    ("qp_mean", "profiles") => w_to_q ∘ g_to_kg, # total precipitation (rain + snow) , not in TC.jl output
    #
    ("t", "timeseries") => null_func, # handled in process_SOCRATES_Atlas_LES_reference rn. could move/fix later
    ("t", "profiles") => null_func, # don't need here
    ("zf", "profiles") => null_func, # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
    ("zf", "reference") => null_func, # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
    #
    ("qi_mean_sub", "profiles") => perday_to_persec,
    ("qi_mean_dep", "profiles") => perday_to_persec,
    ("ql_mean_cond_evap", "profiles") => perday_to_persec, # not sure this is right cause it says droplets
    #

)

default_derived_data_vars_atlas = Dict{Tuple{String, String}, Function}( #TC.jl (Name, Group) =>  fcn
    ("ql_all_mean", "profiles") =>
        x -> Dict(
            "data" => x.group["profiles"]["ql_mean"] .+ x.group["profiles"]["qr_mean"],
            "dimnames" => NC.dimnames(x.group["profiles"]["ql_mean"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("qi_all_mean", "profiles") =>
        x -> Dict(
            "data" =>
                x.group["profiles"]["qi_mean"] .+ x.group["profiles"]["qs_mean"] .+ x.group["profiles"]["qg_mean"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("qip_mean", "profiles") =>
        x -> Dict(
            "data" => x.group["profiles"]["qs_mean"] .+ x.group["profiles"]["qg_mean"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
)

"""
Function to take the Atlas LES output and convert it to the format the calibration expects
 - subset
 - change names

 # each <flight_number, forcing_type> pair has it's own directory since the model expects the ref_dir to only contain one flight
"""

@info default_derived_data_vars_atlas
function process_SOCRATES_Atlas_LES_reference(;
    flight_numbers::Vector{Int} = flight_numbers,
    forcing_types::Vector{Symbol} = forcing_types,
    data_vars = keys(default_data_vars_atlas), # the variables we need...
    data_vars_rename::Dict{Tuple{String, String}, Union{String, Nothing}} = default_data_vars_atlas,
    truth_dir::String = joinpath(this_dir, "Reference", "Atlas_LES"), # the folder where we store our truth (Atlas LES Data)
    out_dir::String = joinpath(this_dir, "Reference", "Atlas_LES"), # the folder where we store our output data
    overwrite::Bool = true,
    derived_data_vars::Dict{Tuple{String, String}, Function} = default_derived_data_vars_atlas, # variables derived from existing vars 
    var_scalings::Dict{Tuple{String, String}, Function} = default_var_LES_to_TC_scaling,
    z_is_c_or_f::String = "c", # "c" or "f" for whether we take the z or f in the Atlas LES data as the center or face of the grid cell. any zf can create a valid zc but not all zc can create a valid zf... but for the Atlas LES data, taking z as zc does yield a valid zf and matches what TC.jl does...
)
    truth_files = filter(contains(".nc"), readdir(truth_dir))


    outfiles = Dict{Tuple{Int, Symbol}, Union{String, Nothing}}(
        (flight_number, forcing_type) => nothing for flight_number in flight_numbers, forcing_type in forcing_types
    )

    for forcing_type in forcing_types
        forcing_type_short = replace.(string.(forcing_type), "_data" => "")  # just the ERA5 or Obs
        for flight_number in flight_numbers
            @info("Processing flight $flight_number and forcing $forcing_type")
            truth_file = filter(
                x ->
                    contains("RF" * string(flight_number, pad = 2))(x) &
                    contains(lowercase(forcing_type_short))(lowercase(x)),
                truth_files,
            )
            if !isempty(truth_file) # if we have a file
                truth_file = truth_file[] # get the item out
                truth_file = joinpath(out_dir, truth_file)
                # Load the truth data
                truth_data = NC.Dataset(truth_file, "r")
                # apparently in julia there's no way to subset and work on a dataset in memory only so from here on we gotta goto disk already...
                name = "RF" * string(flight_number, pad = 2) * "_" * string(forcing_type)
                outpath = joinpath(out_dir, name, "stats", name * ".nc") # the format CalibrateEDMF.jl expects from TurbulenceConvection.jl
                outfiles[(flight_number, forcing_type)] = outpath
                if !isfile(outpath) || overwrite # only do if we don't have the file or we want to overwrite
                    mkpath(dirname(outpath)) # make the directory if it doesn't exist
                    rm(outpath, force = true) # NCDatasets claims it will overwrite file if it exists but it doesn't so we delete if it exists
                    new_data = NC.Dataset(outpath, "c") # create the new dataset
                    # add the required groups for interplay w/ TC.jl
                    for group in ["profiles", "reference", "timeseries"]
                        NC.defGroup(new_data, group)
                    end
                    # add dimensions
                    for _dim in keys(truth_data.dim)
                        NC.defDim(new_data, _dim, truth_data.dim[_dim]) # truth_data.dim[_dim] gives the size
                    end

                    # add attributes
                    for _attrib in keys(truth_data.attrib)
                        Base.setindex!(new_data.attrib, truth_data.attrib[_attrib], _attrib)
                    end
                    # add variables & rename
                    for _vardef in data_vars
                        _data_var, _group = _vardef
                        _truth_var = data_vars_rename[_vardef]

                        if isnothing(_truth_var)
                            continue # this variable doesn't exist in the truth data, so we can't create it without knowing its dimensions etc...
                        end



                        if z_is_c_or_f == "f"
                            _new_dims = collect((x == "z" ? "zf" : x for x in NC.dimnames(truth_data[_truth_var]))) # replace any "z" in dimnames with "zf" (really it's zc in TC.jl but we can't reliaby ensure that any vector zf yields a valid zf w/ z[1] = 0)
                        else
                            _new_dims = collect((x == "z" ? "zc" : x for x in NC.dimnames(truth_data[_truth_var]))) # replace any "z" in dimnames with "zc" # this is technically what TC non advective variable are on...
                        end

                        _truth_vardata = Array(truth_data[_truth_var])

                        # apply prescribed scalings
                        if _vardef in keys(var_scalings)
                            _truth_vardata = var_scalings[_vardef](_truth_vardata)
                        end

                        NC.defVar(
                            new_data.group[_group],
                            _data_var,
                            _truth_vardata,
                            _new_dims;
                            attrib = truth_data[_truth_var].attrib,
                        )

                    end

                    # -- --  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- #
                    #  Calculate and Fixe Variables
                    # -- --  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- #

                    # Calculate and add ("zc", "profiles"), and ("zc", "reference"), grid shouldn't have to be the same as TC.jl and it should interpolate using vertical_interpolation() from HelperFuncs.jl, if so we need to add z=0 to z/zf and also something at the surface for variables defined on z/zf, the latter part being the hard part lol so I hope we don't have to do that...

                    if z_is_c_or_f == "f"
                        if "zf" ∈ keys(new_data.group["profiles"])
                            zf_data = Array(new_data.group["profiles"]["zf"])
                        elseif "zf" ∈ keys(new_data.group["reference"])
                            zf_data = Array(new_data.group["reference"]["zf"])
                        else
                            error(
                                "No `zf` variable found in new_data group `profile`s or `reference``, ensure it is in the created data_vars",
                            )
                        end

                        zc_data = (zf_data[1:(end - 1)] .+ zf_data[2:end]) ./ 2.0 # zf -> zc
                        _new_dims = collect((x == "z" ? "zc" : x for x in NC.dimnames(truth_data["z"]))) # replace any "z" in dimnames with "zc"
                        NC.defVar(
                            new_data.group["profiles"],
                            "zc",
                            zc_data,
                            (_new_dims);
                            attrib = truth_data["z"].attrib,
                        )
                        NC.defVar(
                            new_data.group["reference"],
                            "zc",
                            zc_data,
                            _new_dims;
                            attrib = truth_data["z"].attrib,
                        )
                    else
                        if "zc" ∈ keys(new_data.group["profiles"])
                            zc_data = Array(new_data.group["profiles"]["zc"])
                        elseif "zc" ∈ keys(new_data.group["reference"])
                            zc_data = Array(new_data.group["reference"]["zc"])
                        else
                            error(
                                "No `zc` variable found in new_data group `profile`s or `reference``, ensure it is in the created data_vars",
                            )
                        end

                        zf_data = FT[FT(0);] # zc -> zf
                        for zc in zc_data
                            append!(zf_data, 2 * zc - zf_data[end]) # zf_data[end] + 2*(zc - zf_data[end]) = 2*zc - zf_data[end] # This happens to work for the grids they gave us and not yield any negative numbers...
                        end
                        NC.defVar(new_data.group["profiles"], "zf", zf_data, ["zf"]; attrib = truth_data["z"].attrib)
                        NC.defVar(new_data.group["reference"], "zf", zf_data, ["zf"]; attrib = truth_data["z"].attrib)
                    end




                    # Fix time variable to go from days to second
                    new_data.group["profiles"]["t"][:] =
                        (new_data.group["profiles"]["t"][:] .- new_data.group["profiles"]["t"][1]) .*
                        (24.0 .* 3600.0) |> (x -> x .+ (ceil(x[end] / 3600) * 3600 - x[end])) # shift day to second, then i think t=0 is missing and we need to shift it to end at t=12,14, otherwise would end at 11.91,13.91h so shift to nearest hour
                    new_data.group["timeseries"]["t"][:] =
                        (new_data.group["timeseries"]["t"][:] .- new_data.group["timeseries"]["t"][1]) .*
                        (24.0 .* 3600.0) |> (x -> x .+ (ceil(x[end] / 3600) * 3600 - x[end])) # shift day to second, then i think t=0 is missing and we need to shift it to end at t=12,14, otherwise would end at 11.91,13.91h so shift to nearest hour

                    # Add derived variables
                    for _vardef in keys(derived_data_vars)
                        _data_var, _group = _vardef
                        _new_data = derived_data_vars[_vardef](new_data)
                        NC.defVar(new_data.group[_group], _data_var, _new_data["data"], _new_data["dimnames"]) # drop attributes cause they'd be innacurate
                    end

                    # consider calculating θ_{li} here, so we don't have to use say, temperature_mean in calibration... (probably requires adding Thermodynamics.jl)

                    #close file at end
                    close(new_data)
                else
                    @info(
                        "Skiping existing file for flight $flight_number and forcing $forcing_type since overwrite is false"
                    )
                end
            else
                @info(
                    "Skiping creating reference due to missing truth file for flight $flight_number and forcing $forcing_type"
                )
            end
        end
    end

    return outfiles
end


# The variables we need for calibration routine from the Atlas LES outputs
default_data_vars_obs = Dict{Tuple{String, String}, String}( #TC.jl (Name, Group) =>  Obs Name (currently matches "Research_Schneider/Projects/Microphysics/Data/SOCRATES/Atlas_LES_Profiles/Reference_Profiles/Atlas_Binned_Profiles/RF_all__leg_types_U_D_min_time_120__2_mb_binning__binned_profiles_z.nc")
    ("temperature_mean", "profiles") => "T",
    ("qt_mean", "profiles") => "qt",
    ("ql_mean", "profiles") => "ql_incloud", # Seems QCL and QC are the same
    ("qi_mean", "profiles") => "qi_incloud", # seems QCI and QI are the same
    ("qc_mean", "profiles") => "qc_incloud", # cloud liquid and ice (not in TC.jl output)
    # ("qr_mean", "profiles") => "RWC", # Haven't added to binning yet
    # ("qs_mean", "profiles") => "", # can't find a variable for
    ("zf", "profiles") => "z", # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
    ("zf", "reference") => "z", # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
)

default_derived_data_vars_obs = Dict{Tuple{String, String}, Function}()

#= 
# This has no time dimension and we need to take the median of the profiles, how to fix?
    For calibrations, we would need a time axis, mayb we can:
    - convert profiles to time and set the time coordinates to linear space between t_start and t_end of the reference period
    - use the LES for covariances but the truth be the median of the LES profiles (check how to do...)
=#

"""
Function to take the SOCRATES Flight Obs output and convert it to the format the calibration expects
 - subset
 - change names

 # each <flight_number, forcing_type> pair has it's own directory since the model expects the ref_dir to only contain one flight

 I think if we use fake_time_axis, we don't need time_shift to change from normal atlas runs just cause we don't start at 0 and have the same time values...
"""
function process_SOCRATES_Flight_Observations_reference(;
    flight_numbers::Vector{Int} = flight_numbers,
    forcing_types::Vector{Symbol} = forcing_types,
    data_vars = keys(default_data_vars_obs), # the variables we need...
    data_vars_rename::Dict{Tuple{String, String}, Union{String, Nothing}} = default_data_vars_obs,
    truth_file::String = joinpath(
        this_dir,
        "Reference",
        "Flight_Observations",
        "RF_all__leg_types_U_D_min_time_120__2_mb_binning__binned_profiles_z.nc",
    ),  # If you wanna rebin yourself, then you'd have to tie into my python routines w/ pycall or something...
    fake_time_axis::Bool = true, # if true, we will create a time axis from the profiles and set the time coordinates to linear space between t_start and t_end of the reference period
    fake_time_bnds = "Atlas",
    out_dir::Union{String, Nothing} = nothing,
    out_vars = nothing, # if not nothing, we will only output these variables and drop any rows for which any of the varialbes are all nan or only one value (or the intersection between variables that is not nan has only one variable)
    overwrite::Bool = true,
    derived_data_vars::Dict{Tuple{String, String}, Function} = default_derived_data_vars_obs, # variables derived from existing vars 
)

    if !isempty(truth_file) # if we have a file
        truth_data = NC.Dataset(truth_file, "r")
    else
        error("No truth file found")
    end

    if isnothing(out_dir)
        if fake_time_axis
            out_dir = joinpath(this_dir, "Reference", "Flight_Observations", "Faked_Profiles_to_Time")
        else
            out_dir = joinpath(this_dir, "Reference", "Flight_Observations")
        end
    end

    # create directory to store save path for each flight_number and forcing type prealloacated to already be filled with nothing by comprehension
    outfiles = Dict{Tuple{Int, Symbol}, Union{String, Nothing}}(
        (flight_number, forcing_type) => nothing for flight_number in flight_numbers, forcing_type in forcing_types
    )
    outfiles_trimmed_outvars = Dict{Tuple{Int, Symbol}, Union{String, Nothing}}(
        (flight_number, forcing_type) => nothing for flight_number in flight_numbers, forcing_type in forcing_types
    )

    for forcing_type in forcing_types
        # forcing_type_short =  replace.(string.(forcing_type), "_data" => "")  # just the ERA5 or Obs
        for flight_number in flight_numbers
            @info("Processing flight $flight_number and forcing $forcing_type")

            _truth_data = NC.@select(truth_data, flight_number == $flight_number) # HOW TO SQUEEZE? 
            # return _truth_data
            # _truth_data = truth_data

            # apparently in julia there's no way to subset and work on a dataset in memory only so from here on we gotta goto disk already...
            name = "RF" * string(flight_number, pad = 2) * "_" * string(forcing_type)
            outpath = joinpath(out_dir, name, "stats", name * ".nc") # the format CalibrateEDMF.jl expects from TurbulenceConvection.jl
            outfiles[(flight_number, forcing_type)] = outpath

            if !isfile(outpath) || overwrite # only do if we don't have the file or we want to overwrite

                mkpath(dirname(outpath)) # make the directory if it doesn't exist
                rm(outpath, force = true) # NCDatasets claims it will overwrite file if it exists but it doesn't so we delete if it exists
                new_data = NC.Dataset(outpath, "c") # create the new dataset
                # add the required groups for interplay w/ TC.jl
                for group in ["profiles", "reference", "timeseries"]
                    NC.defGroup(new_data, group)
                end
                # add dimensions

                new_dims = filter(x -> x != "flight_number", keys(_truth_data.dim)) # cause we can't squeeze flight_number dimension after using @select

                if new_dims == ["profile", "z"]
                    new_dims = ["z", "profile"] # we need to swap the order because CalibrateEDMF.jl for some reason assumes "z" has to be second (which is first in the netcdf bc julia is column major but netcdf/c is row major, see https://github.com/Alexander-Barth/NCDatasets.jl/issues/87#issuecomment-636098859)
                end

                for _dim in new_dims # CAUSE WE CAN'T SQUEEZE
                    NC.defDim(new_data, _dim, _truth_data.dim[_dim]) # _truth_data.dim[_dim] gives the size
                end

                # add attributes
                for _attrib in keys(_truth_data.attrib)
                    Base.setindex!(new_data.attrib, truth_truth_data_data.attrib[_attrib], _attrib)
                end
                # add variables & rename
                for _vardef in data_vars
                    _data_var, _group = _vardef
                    _truth_var = data_vars_rename[_vardef]
                    _new_dims = collect((x == "z" ? "zf" : x for x in NC.dimnames(_truth_data[_truth_var]))) # replace any "z" in dimnames with "zf"

                    flight_number_ind = findfirst(x -> x == "flight_number", _new_dims) # find the index of the flight_number dimension since we couldn't squeeze it out
                    _new_dims = filter(x -> x != "flight_number", _new_dims) # CAUSE WE CAN'T SQUEEZE
                    _new_data = Array(_truth_data[_truth_var])

                    if !isnothing(flight_number_ind)
                        _new_data = dropdims(_new_data; dims = flight_number_ind) # drop flight_number dimension since we cant squeeze
                    end

                    if _new_dims == ["profile", "zf"]
                        _new_dims = ["zf", "profile"] # we need to swap the order because CalibrateEDMF.jl for some reason assumes "z" has to be second (which is first in the netcdf bc julia is column major but netcdf/c is row major, see https://github.com/Alexander-Barth/NCDatasets.jl/issues/87#issuecomment-636098859)
                        zf_dim = findfirst(x -> x == "zf", _new_dims)
                        profile_dim = findfirst(x -> x == "profile", _new_dims)
                        _new_data = permutedims(
                            _new_data,
                            [i == zf_dim ? profile_dim : (i == profile_dim ? zf_dim : i) for i in 1:ndims(_new_data)],
                        ) # swapdims for zf_dim and t_dim
                    end


                    NC.defVar(
                        new_data.group[_group],
                        _data_var,
                        _new_data,
                        _new_dims;
                        attrib = _truth_data[_truth_var].attrib,
                    )
                end

                # -- --  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- #
                #  Calculate and Fix Variables
                # -- --  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- #

                # Calculate and add ("zc", "profiles"), and ("zc", "reference"), grid shouldn't have to be the same as TC.jl and it should interpolate using vertical_interpolation() from HelperFuncs.jl, if so we need to add z=0 to z/zf and also something at the surface for variables defined on z/zf, the latter part being the hard part lol so I hope we don't have to do that...
                zf_data = Array(new_data.group["profiles"]["zf"])
                zc_data = (zf_data[1:(end - 1)] .+ zf_data[2:end]) ./ 2.0
                _new_dims = collect((x == "z" ? "zc" : x for x in NC.dimnames(_truth_data["z"]))) # replace any "z" in dimnames with "zc"

                NC.defVar(new_data.group["profiles"], "zc", zc_data, _new_dims; attrib = _truth_data["z"].attrib)
                NC.defVar(new_data.group["reference"], "zc", zc_data, _new_dims; attrib = _truth_data["z"].attrib)

                # consider calculating θ_{li} here, so we don't have to use say, temperature_mean in calibration... (probably requires adding Thermodynamics.jl)

                # Add derived variables
                for _vardef in keys(derived_data_vars)
                    _data_var, _group = _vardef
                    _data_varfunc = derived_data_vars[_vardef]
                    _new_data, new_dims = _data_varfunc(new_data)
                    NC.defVar(
                        new_data.group[_group],
                        _data_varfunc,
                        _new_data,
                        _new_dims;
                        attrib = _truth_data[_new_data].attrib,
                    )
                end


                # -- --  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- #
                #  Turn profile into a `time` axis, so we can use it in the CalibrateEDMF.jl pipeline
                # -- --  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- #

                if fake_time_axis
                    if fake_time_bnds == "Atlas"
                        # get our start and end time
                        if forcing_type == :obs_data # From Atlas paper, hour 10-12 are used for comparing obs
                            t_start = 10 * 3600.0
                            t_end = 12 * 3600.0
                        elseif forcing_type == :ERA5_data # Use the start and end times from Table 2 in atlas, stored in SOCRATES_summary.nc that we created w/ a Python Jupyter notebook
                            NC.Dataset(joinpath(this_dir, "Reference", "SOCRATES_summary.nc"), "r") do SOCRATES_summary
                                _sum = NC.@select(SOCRATES_summary, flight_number == $flight_number)
                                t_start, t_end = _sum["time_bnds"]
                                t_ref = _sum["reference_time"][1] # we know this is hour 12
                                t_start, t_end =
                                    map(x -> x.value, Dates.Second.([t_start, t_end] .- t_ref)) .+ (12 * 3600) # get the difference in seconds between t_start,t_end and t_ref = 12 hours, and add to the 12 hours to get the final values in seconds
                            end
                        end # SOCRATES_summary is closed
                    else
                        t_start, t_end = fake_time_bnds # allow us to fake a different time range, e.g. for testing shorter simulations
                    end
                    fake_time_bnds_str = string(t_start, "_", t_end)


                    t = collect(range(t_start, t_end, length = new_data.dim["profile"]))
                    NC.renameDim(new_data, "profile", "t")  # renmae profiles to "t" undocumented, see https://github.com/Alexander-Barth/NCDatasets.jl/blob/master/src/dimensions.jl
                    NC.renameDim(new_data.group["profiles"], "profile", "t")  # renmae profiles to "t" undocumented, see https://github.com/Alexander-Barth/NCDatasets.jl/blob/master/src/dimensions.jl
                    NC.defVar(new_data.group["profiles"], "t", t, ["t"])
                    NC.defVar(new_data.group["reference"], "t", t, ["t"])
                    NC.defVar(new_data.group["timeseries"], "t", t, ["t"])
                end

            else
                @info(
                    "Skiping overwriting existing file for flight $flight_number and forcing $forcing_type since overwrite is false"
                )
                new_data = NC.Dataset(outpath, "r")
            end

            # drop empty zf_inds and zf_inds with just one value (in time) so that we can calculate covariances w/o just getting NaN
            valid_zf_inds = collect(range(1, size(new_data.group["profiles"]["zf"])[1], step = 1)) # all zf_inds are valid initially
            # @info(valid_zf_inds)
            if isnothing(out_vars)
                out_vars = keys(new_data.group["profiles"])
                # filter to 2d variables
                out_vars = filter(x -> length(NC.dimnames(new_data.group["profiles"][x])) == 2, out_vars)
                # @info(out_vars)
            end
            # @info(out_vars)
            for zf_ind in deepcopy(valid_zf_inds)
                # @info("zf_ind $zf_ind")
                # zf_inds fewer than or equal to 1 non-nan value
                for var in out_vars
                    # @info(var, zf_ind,)
                    # @info(size(Array(new_data.group["profiles"][var])))
                    if sum((!isnan).(replace(new_data.group["profiles"][var][zf_ind, :], missing => NaN))) <= 1
                        deleteat!(valid_zf_inds, findfirst(x -> x == zf_ind, valid_zf_inds))
                        # @info("Deleting zf_ind $zf_ind (zf = $(new_data.group["profiles"]["zf"][zf_ind])) it has fewer than or equal to 1 non-nan value for variable $var")
                        # @info("deleted less than 1, valid_zf_inds: $valid_zf_inds")
                        break
                    end
                    # zf_inds with some pair of variables having intersection of non-nan values of length 1 or less
                    for var2 in out_vars
                        if var != var2
                            # @info( hcat(new_data.group["profiles"][var][zf_ind,:], new_data.group["profiles"][var2][zf_ind,:]))
                            if length(
                                intersect(
                                    findall(
                                        (!isnan).(replace(new_data.group["profiles"][var][zf_ind, :], missing => NaN)),
                                    ),
                                    findall(
                                        (!isnan).(replace(new_data.group["profiles"][var2][zf_ind, :], missing => NaN)),
                                    ),
                                ),
                            ) <= 1
                                deleteat!(valid_zf_inds, findfirst(x -> x == zf_ind, valid_zf_inds))
                                # @info("Deleting zf_ind $zf_ind (zf=$(new_data.group["profiles"]["zf"][zf_ind])) because the intersection of non-nan values for $var and $var2 is of length 1 or less, meaning NaNs in the covariance matrix")
                                # @info("deleted intersection of less than 1, valid_zf_inds: $valid_zf_inds")
                                @goto break_out_both_for_loops
                            end
                        end
                    end
                end
                @label break_out_both_for_loops
                # @info("loop end, valid_zf_inds: $valid_zf_inds")
            end
            @info("Flight number $flight_number, forcing $forcing_type, valid zf_inds: $valid_zf_inds")

            if length(valid_zf_inds) > 0
                # @info (new_data)
                valid_zfs = new_data.group["profiles"]["zf"][valid_zf_inds]
                # @info(valid_zfs)
                # @info(new_data.group["profiles"]["zf"])

                # save to file, appending which out_vars we kept separated by underscores
                outpath_outvars = joinpath(
                    out_dir,
                    name,
                    "stats",
                    "$(join(out_vars, "_"))" * "_time_" * fake_time_bnds_str,
                    name * ".nc",
                ) # the format CalibrateEDMF.jl expects from TurbulenceConvection.jl
                outfiles_trimmed_outvars[(flight_number, forcing_type)] = outpath_outvars
                if !isfile(outpath_outvars) || overwrite

                    # save new_data_2 to outpath_outvars
                    mkpath(dirname(outpath_outvars)) # make the directory if it doesn't exist
                    rm(outpath_outvars, force = true) # NCDatasets claims it will overwrite file if it exists but it doesn't so we delete if it exists
                    new_data_2 = NC.Dataset(outpath_outvars, "c") # create the new dataset
                    # # add dims


                    # for _dim in keys(new_data_2.dim)
                    #     NC.defDim(new_data_2, _dim, new_data_2.dim[_dim]) # new_data_2.dim[_dim] gives the size
                    # end

                    # add attributes
                    for _attrib in keys(new_data.attrib)
                        Base.setindex!(new_data_2.attrib, new_data.attrib[_attrib], _attrib)
                    end

                    # add the required groups for interplay w/ TC.jl
                    for group in ["profiles", "reference", "timeseries"]
                        NC.defGroup(new_data_2, group)

                        if "zf" in NC.dimnames(new_data.group[group])
                            _new_data_2 = NC.@select(new_data.group[group], zf ∈ $valid_zfs) # select only valid zf_inds (need to do by group?)
                        else
                            _new_data_2 = new_data.group[group]
                        end

                        # @info("_new_data_2", _new_data_2)

                        # @info(_data_vars, _new_dims)

                        # add Variables
                        _data_vars = keys(_new_data_2)
                        for var in _data_vars
                            # if (var in out_vars) || (var in ["t", "zf", "zc"]) # only add the variables we want to keep, or dimensions/coordinates 
                            if true # keep all variables despite having trimmed on out_vars, just for future reference
                                #
                                # @info(var, keys(_new_data_2))
                                _new_dims = NC.dimnames(_new_data_2[var])
                                NC.defVar(
                                    new_data_2.group[group],
                                    var,
                                    Array(_new_data_2[var]),
                                    _new_dims;
                                    attrib = _new_data_2[var].attrib,
                                )
                            end
                        end
                    end
                    @info("closing")
                    close(new_data_2)
                else
                    @info(
                        "Skiping overwriting existing trimmed file for flight $flight_number and forcing $forcing_type since overwrite is false"
                    )
                end
            end


            #close file at end
            close(new_data)

        end
    end

    return outfiles, outfiles_trimmed_outvars

end
