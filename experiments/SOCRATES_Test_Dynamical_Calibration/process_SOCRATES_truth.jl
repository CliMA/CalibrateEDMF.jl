using CalibrateEDMF
const NC = CalibrateEDMF.NetCDFIO.NC # is already loaded so let's not reload
flight_numbers = [1,9,10,11,12,13]
forcing_types  = [:obs_data, :ERA5_data]

# this_dir = @__DIR__ # the location of this file
this_dir = "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Test_Dynamical_Calibration/"
data_dir = joinpath(this_dir, "Truth") # the folder where we store our truth (Atlas LES Data)

# The variables we need for calibration routine from the Atlas LES outputs
default_data_vars = Dict{Tuple{String,String}, String}( #TC.jl (Name, Group) =>  Atlas Name
("thetal_mean", "profiles") => "THETAL",
("temperature_mean", "profiles") => "TABS",
("qt_mean", "profiles") => "QT",
("ql_mean", "profiles") => "QCL", # Seems QCL and QC are the same
("qi_mean", "profiles") => "QCI", # seems QCI and QI are the same
("qc_mean", "profiles") => "QN", # cloud liquid and ice (not in TC.jl output)
("qr_mean", "profiles") => "QR",
("qs_mean", "profiles") => "QS",
("qp_mean", "profiles") => "QP", # total precipitation (rain + snow) , not in TC.jl output
("t", "timeseries") => "time",
("t", "profiles") => "time",
("zf", "profiles") => "z", # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
("zf", "reference") => "z", # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
)

"""
Function to take the Atlas LES output and convert it to the format the calibration expects
 - subset
 - change names

 # each <flight_number, forcing_type> pair has it's own directory since the model expects the ref_dir to only contain one flight
"""
function process_SOCRATES_truth(;
    flight_numbers::Vector{Int} = flight_numbers,
    forcing_types::Vector{Symbol} = forcing_types,
    data_vars = keys(default_data_vars), # the variables we need...
    data_vars_rename::Dict{Tuple{String,String}, String} = default_data_vars,
    )
    truth_files = filter(contains(".nc",), readdir(data_dir))

    for forcing_type in forcing_types
        forcing_type_short =  replace.(string.(forcing_type), "_data" => "")  # just the ERA5 or Obs
        for flight_number in flight_numbers
            @info("Processing flight $flight_number and forcing $forcing_type")
            truth_file = filter(x -> contains("RF"*string(flight_number,pad=2))(x) & contains(lowercase(forcing_type_short))(lowercase(x)), truth_files)
            if ~isempty(truth_file) # if we have a file
                truth_file = truth_file[] # get the item out
                truth_file = joinpath(data_dir, truth_file)
                # Load the truth data
                truth_data = NC.Dataset(truth_file,"r")
                # apparently in julia there's no way to subset and work on a dataset in memory only so from here on we gotta goto disk already...
                name = "RF"*string(flight_number,pad=2)*"_"*string(forcing_type)
                outpath = joinpath(data_dir, name, "stats",name*".nc") # the format CalibrateEDMF.jl expects from TurbulenceConvection.jl
                mkpath(dirname(outpath)) # make the directory if it doesn't exist
                rm(outpath,force=true) # NCDatasets claims it will overwrite file if it exists but it doesn't so we delete if it exists
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
                    _new_dims = collect((x=="z" ? "zf" : x for x in NC.dimnames(truth_data[_truth_var]))) # replace any "z" in dimnames with "zf"
                    NC.defVar(new_data.group[_group], _data_var, truth_data[_truth_var][:], _new_dims; attrib =  truth_data[_truth_var].attrib )
                end

                # -- --  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- #
                #  Calculate and Fixe Variables
                # -- --  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- #

                # Calculate and add ("zc", "profiles"), and ("zc", "reference"), grid shouldn't have to be the same as TC.jl and it should interpolate using vertical_interpolation() from HelperFuncs.jl, if so we need to add z=0 to z/zf and also something at the surface for variables defined on z/zf, the latter part being the hard part lol so I hope we don't have to do that...
                zf_data = new_data.group["profiles"]["zf"][:]
                zc_data = (zf_data[1:end-1] .+ zf_data[2:end]) ./ 2.0
                _new_dims  =  collect((x=="z" ? "zc" : x for x in NC.dimnames(truth_data["z"]))) # replace any "z" in dimnames with "zc"
                NC.defVar(new_data.group["profiles"], "zc", zc_data, (_new_dims); attrib =  truth_data["z"].attrib )
                NC.defVar(new_data.group["reference"], "zc", zc_data, _new_dims; attrib =  truth_data["z"].attrib )
                # Fix time variable to go from days to second
                new_data.group["profiles"]["t"][:] = (new_data.group["profiles"]["t"][:] .- new_data.group["profiles"]["t"][1]) .* (24.0 .* 3600.0) |> (x -> x .+ (ceil(x[end]/3600)*3600 - x[end])) # shift day to second, then i think t=0 is missing and we need to shift it to end at t=12,14, otherwise would end at 11.91,13.91h so shift to nearest hour
                new_data.group["timeseries"]["t"][:] = (new_data.group["timeseries"]["t"][:] .- new_data.group["timeseries"]["t"][1]) .* (24.0 .* 3600.0 ) |> (x -> x .+ (ceil(x[end]/3600)*3600 - x[end])) # shift day to second, then i think t=0 is missing and we need to shift it to end at t=12,14, otherwise would end at 11.91,13.91h so shift to nearest hour
                # Fix water to go from g/kg = kg/kg
                new_data.group["profiles"]["qt_mean"][:] = new_data.group["profiles"]["qt_mean"][:] ./ 1000.0
                new_data.group["profiles"]["ql_mean"][:] = new_data.group["profiles"]["ql_mean"][:] ./ 1000.0
                new_data.group["profiles"]["qi_mean"][:] = new_data.group["profiles"]["qi_mean"][:] ./ 1000.0

                # consider calculating θ_{li} here, so we don't have to use say, temperature_mean in calibration... (probably requires adding Thermodynamics.jl)
                
                #close file at end
                close(new_data)
            else
                @info("Skiping missing file for flight $flight_number and forcing $forcing_type")
            end
        end
    end
end

