#= Custom calibration configuration file. =#

using Distributions
using StatsBase
using LinearAlgebra
using Random
using CalibrateEDMF
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.LESUtils
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.HelperFuncs
# Import EKP modules
using JLD2
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const NC = CalibrateEDMF.NetCDFIO.NC # is already loaded so let's not reload

# Cases defined as structs for quick access to default configs
struct SOCRATES_Train end
struct SOCRATES_Val end
this_dir = @__DIR__ # the location of this file
data_dir = joinpath(this_dir, "Truth") # the folder where we store our truth (Atlas LES Data)
flight_numbers = [1,9,10,11,12,13]
forcing_types  = [:obs_data, :ERA5_data]

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(SOCRATES_Train())
    # Define reference used for validation
    # config["validation"] = get_reference_config(SOCRATES_Val()) # No validation for now, until we maybe try calibrating everything together and then validating on a different flight
    # Define the parameter priors
    config["prior"] = get_prior_config()
    # Define the kalman process
    config["process"] = get_process_config()
    # Define the SCM static configuration
    config["scm"] = get_scm_config()
    return config
end

function get_output_config()
    config = Dict()
    config["outdir_root"] = joinpath(this_dir, "Output") # store them in the experiment folder here by default
    return config
end

function get_regularization_config()
    config = Dict()
    # Regularization of observations: mean and covariance
    config["perform_PCA"] = true # Performs PCA on data
    config["variance_loss"] = 1.0e-2 # Variance truncation level in PCA
    config["normalize"] = true  # whether to normalize data by pooled variance
    config["tikhonov_mode"] = "relative" # Tikhonov regularization
    config["tikhonov_noise"] = 1.0e-6 # Tikhonov regularization
    config["dim_scaling"] = true # Dimensional scaling of the loss

    # Parameter regularization: L2 regularization with respect to prior mean.
    #  - Set to `nothing` to use prior covariance as regularizer,
    #  - Set to a float for isotropic parameter regularization.
    #  - Pass a dictionary of lists similar to config["prior_mean"] for
    #       anisotropic regularization. The dictionary must be complete.
    #       If you want to avoid regularizing a certain parameter, set the entry
    #       to [0].
    # To turn off regularization, set config["process"]["augmented"] to false.
    #
    #
    # Defaults set to batch_size/total_size to match total dataset uncertainty
    # in UKI. Feel free to set treat these as hyperparameters.
    config["l2_reg"] = Dict(
        # entrainment parameters
        # "nn_ent_params" => repeat([0.0], 58),
        "turbulent_entrainment_factor" => [5.0 / 60.0],

        # diffusion parameters
        "tke_ed_coeff" => [5.0 / 60.0],
        "tke_diss_coeff" => [5.0 / 60.0],
        "static_stab_coeff" => [5.0 / 60.0],
        "tke_surf_scale" => [5.0 / 60.0],
        "Prandtl_number_0" => [5.0 / 60.0],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [5.0 / 60.0],
        "pressure_normalmode_buoy_coeff1" => [5.0 / 60.0],
        "pressure_normalmode_drag_coeff" => [5.0 / 60.0],

        # surface
        "surface_area" => [5.0 / 60.0],
    )
    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 10 # reduced temporarily for testing
    config["N_ens"] = 10 # Must be 2p+1 when algorithm is "Unscented"
    config["algorithm"] = "Inversion" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = 60.0 / 5.0 # 1.0
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = true
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    return config
end

function get_reference_config(::SOCRATES_Train)
    config = Dict()

    # Setup SOCRATES run arguments I guess
    flight_numbers = [13,]
    forcing_types = [:obs_data,]
    aux_kwargs = () # fill in later

    setups = collect(Iterators.product(flight_numbers, forcing_types))[:]
    setups = map(x -> Dict("flight_number" => x[1], "forcing_type" => x[2]), setups) # convert to list of dictionaries
    # add this here because it seems to be called?
    for setup in setups
       name = "RF"*string(setup["flight_number"],pad=2)*"_"*string(setup["forcing_type"])
       datafile = joinpath(this_dir, "Truth", name, "stats", name*".nc")
       if isfile(datafile)
           setup["datafile"] = datafile
           setup["case_name"] = "SOCRATES_"*name
       else
           @warn("File $datafile does not exist")
       end
    end

    NC.Dataset(joinpath(data_dir, "SOCRATES_summary.nc"),"r") do SOCRATES_summary
        for setup in setups # set up the periods we take our means over to match atlas (no idea what to do about the covariances, maybe just take the same values?)
            if setup["forcing_type"] == :obs_data # From Atlas paper, hour 10-12 are used for comparing obs
                setup["t_start"] = 10 * 3600.
                setup["t_end"]   = 12 * 3600.
            elseif setup["forcing_type"] == :ERA5_data # Use the start and end times from Table 2 in atlas, stored in SOCRATES_summary.nc that we created w/ a Python Jupyter notebook
            _sum = NC.@select(SOCRATES_summary, flight_number == $flight_number)
                t_start, t_end =  _sum["time_bnds"]
                t_ref =  _sum["reference_time"][1] # we know this is hour 12
                t_start, t_end =  map( x-> x.value, Dates.Second.([t_start,t_end] .- t_ref)) .+ (12 * 3600) # get the difference in seconds between t_start,t_end and t_ref = 12 hours, and add to the 12 hours to get the final values in seconds
                setup["t_start"] = t_start
                setup["t_end"] = t_end
            else
                error("invalid forcing_type: ", setup["forcing_type"])
            end
        end
    end # SOCRATES_summary is closed

    setups = filter(d -> haskey(d,"datafile"), setups) # filter out if datafile doesn't exist (11 :obsdata for example)
    n_repeat = length(setups)
    ref_dirs = [dirname(setup["datafile"]) for setup in setups]
    
    # need reference dirs from wherever i put my truth, maybe add a SOCRATESUtils to match les_utils etc.
    n_repeat = length(ref_dirs)
    config["case_name"] = [setup["case_name"] for setup in setups]
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = LES() # our reference is atlas LES
    config["Σ_reference_type"] = LES()
    calibrate_vars =  unique(map(x->x[1], collect(keys(default_data_vars)))) # unique to remove duplicate from different netcdf groups
    config["y_names"] = repeat([calibrate_vars], n_repeat) # the variable we want to calibrate on
    config["y_dir"] = ref_dirs
    # config["t_start"] = repeat([setup["t_start"]], n_repeat) # I think should be 10 hr for Obs, and from Table 2 Atlas for ERA5
    # config["t_end"] = repeat([setup["t_end"]], n_repeat) # I think should be 12 hr for Obs, and from Table 2 Atlas for ERA5
    config["t_start"] = repeat([1800], n_repeat) # shorter for test
    config["t_end"] = repeat([3600], n_repeat) # shorter for testing
    # Use full  timeseries for covariance (for us we just use what)? # for covariance
    # config["Σ_t_start"] = repeat([11.0 * 3600], n_repeat) #  use hours 11-13 for comparison
    # config["Σ_t_end"] = repeat([13.0 * 3600], n_repeat)  #  use hours 11-13 for comparison
    config["Σ_t_start"] = repeat([2500], n_repeat) #  shorter for testing
    config["Σ_t_end"] = repeat([2*3600], n_repeat)  #  shorter for testing (spanning at least 600, our output frequency)
    config["time_shift"] = [setup["forcing_type"] == :obs_data ? 12 * 3600. : 14 * 3600. for setup in setups] # The shift is essentially how far back from the end in LES data does the TC data start. Here they start at the same place so it's the full length of the ATLAS LES model (12 for obs, 14 for era), must also be float type
    # config["time_shift"] = config["time_shift"][] # test to see if vector here was the problem
    # config["batch_size"] = n_repeat # has to be some divisor of n_repeat, default is n_repeat == length(ref_dirs) == number of setups
    config["write_full_stats"] = false
    return config
end

function get_reference_config(::SOCRATES_Val)
    config = Dict()

    # Train on same thing? or waht do we do here      

    flight_numbers = [13,] # just validate on same flight
    forcing_types = [:obs_data,]
    aux_kwargs     = () # fill in later

    setups = collect(Iterators.product(flight_numbers, forcing_types))[:]
    setups = map(x -> Dict("flight_number" => x[1], "forcing_type" => x[2]), setups) # convert to list of dictionaries
    # add this here because it seems to be called?
    for setup in setups
       name = "RF"*string(setup["flight_number"],pad=2)*"_"*string(setup["forcing_type"])
       datafile = joinpath(this_dir, "Truth", name, "stats", name*".nc")
       if isfile(datafile)
           setup["datafile"] = datafile
           setup["case_name"] = "SOCRATES_"*name
       else
           @warn("File $datafile does not exist")
       end
    end

    NC.Dataset(joinpath(data_dir, "SOCRATES_summary.nc"),"r") do SOCRATES_summary
        for setup in setups # set up the periods we take our means over to match atlas (no idea what to do about the covariances, maybe just take the same values?)
            if setup["forcing_type"] == :obs_data # From Atlas paper, hour 10-12 are used for comparing obs
                setup["t_start"] = 10 * 3600.
                setup["t_end"]   = 12 * 3600.
            elseif setup["forcing_type"] == :ERA5_data # Use the start and end times from Table 2 in atlas, stored in SOCRATES_summary.nc that we created w/ a Python Jupyter notebook
            _sum = NC.@select(SOCRATES_summary, flight_number == $flight_number)
                t_start, t_end =  _sum["time_bnds"]
                t_ref =  _sum["reference_time"][1] # we know this is hour 12
                t_start, t_end =  map( x-> x.value, Dates.Second.([t_start,t_end] .- t_ref)) .+ (12 * 3600) # get the difference in seconds between t_start,t_end and t_ref = 12 hours, and add to the 12 hours to get the final values in seconds
                setup["t_start"] = t_start
                setup["t_end"] = t_end
            else
                error("invalid forcing_type: ", setup["forcing_type"])
            end
        end
    end # SOCRATES_summary is closed

    setups = filter(d -> haskey(d,"datafile"), setups) # filter out if datafile doesn't exist (11 :obsdata for example)
    n_repeat = length(setups)
    ref_dirs = [dirname(setup["datafile"]) for setup in setups]

    # need reference dirs from wherever i put my truth, maybe add a SOCRATESUtils to match les_utils etc.
    n_repeat = length(ref_dirs)
    config["case_name"] = [setup["case_name"] for setup in setups]
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = LES() # our reference is atlas LES
    config["Σ_reference_type"] = LES()
    calibrate_vars =  unique(map(x->x[1], collect(keys(default_data_vars)))) # unique to remove duplicate from different netcdf groups
    config["y_names"] = repeat([calibrate_vars], n_repeat) # the variable we want to calibrate on
    config["y_dir"] = ref_dirs
    # config["t_start"] = repeat([setup["t_start"]], n_repeat) # I think should be 10 hr for Obs, and from Table 2 Atlas for ERA5
    # config["t_end"] = repeat([setup["t_end"]], n_repeat) # I think should be 12 hr for Obs, and from Table 2 Atlas for ERA5
    config["t_start"] = repeat([1800], n_repeat) # shorter for test
    config["t_end"] = repeat([3600], n_repeat) # shorter for testing
    # Use full  timeseries for covariance (for us we just use what)? # for covariance
    # config["Σ_t_start"] = repeat([11.0 * 3600], n_repeat) #  use hours 11-13 for comparison
    # config["Σ_t_end"] = repeat([13.0 * 3600], n_repeat)  #  use hours 11-13 for comparison
    config["Σ_t_start"] = repeat([2500], n_repeat) #  shorter for testing
    config["Σ_t_end"] = repeat([2*3600], n_repeat)  #  shorter for testing (spanning at least 600, our output frequency)
    config["time_shift"] = [setup["forcing_type"] == :obs_data ? 12 * 3600. : 14 * 3600. for setup in setups] # The shift is essentially how far back from the end in LES data does the TC data start. Here they start at the same place so it's the full length of the ATLAS LES model (12 for obs, 14 for era), must also be float type
    # config["time_shift"] = config["time_shift"][] # test to see if vector here was the problem
    # config["batch_size"] = n_repeat # has to be some divisor of n_repeat, default is n_repeat == length(ref_dirs) == number of setups
    config["write_full_stats"] = false
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # entrainment parameters
        # "nn_ent_params" => [repeat([no_constraint()], 58)...], # remove
        # need to add entrainment parameters for moisture deficit clousure
        "turbulent_entrainment_factor" => [bounded(0.0, 10.0)],

        # diffusion parameters
        "tke_ed_coeff" => [bounded(0.01, 1.0)],
        "tke_diss_coeff" => [bounded(0.01, 1.0)],
        "static_stab_coeff" => [bounded(0.01, 1.0)],
        "tke_surf_scale" => [bounded(1.0, 16.0)],
        "Prandtl_number_0" => [bounded(0.5, 1.5)],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [bounded(0.0, 100.0)],
        "pressure_normalmode_buoy_coeff1" => [bounded(0.0, 10.0)],
        "pressure_normalmode_drag_coeff" => [bounded(0.0, 50.0)],

        # surface
        "surface_area" => [bounded(0.01, 0.5)],
    )

    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # entrainment parameters
        # "nn_ent_params" => 0.1 .* (rand(58) .- 0.5),
        "turbulent_entrainment_factor" => [0.075],

        # diffusion parameters
        "tke_ed_coeff" => [0.14],
        "tke_diss_coeff" => [0.22],
        "static_stab_coeff" => [0.4],
        "tke_surf_scale" => [3.75],
        "Prandtl_number_0" => [0.74],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [0.001],
        "pressure_normalmode_buoy_coeff1" => [0.12],
        "pressure_normalmode_drag_coeff" => [10.0],

        # surface
        "surface_area" => [0.1],
    )

    config["unconstrained_σ"] = 1.0 # just leave everyting variance 
    # Tight initial prior for Unscented
    # config["unconstrained_σ"] = 0.25
    return config
end

function get_scm_config() # set all my namelist stuff here
    config = Dict()
    config["namelist_args"] = [
        ("time_stepping", "dt_min", 0.5),
        ("time_stepping", "dt_max", 2.0),
        ("time_stepping", "t_max", 14*3600.0),
        ("turbulence", "EDMF_PrognosticTKE", "entrainment", "None"),
        ("turbulence", "EDMF_PrognosticTKE", "ml_entrainment", "NN"),
        ("turbulence", "EDMF_PrognosticTKE", "area_limiter_power", 0.0),
        ("turbulence", "EDMF_PrognosticTKE", "entr_dim_scale", "inv_z"),
        ("turbulence", "EDMF_PrognosticTKE", "detr_dim_scale", "inv_z"),
    ]
    return config
end

# The variables we need for calibration routine from the Atlas LES outputs
default_data_vars = Dict{Tuple{String,String}, String}( #TC.jl (Name, Group) =>  Atlas Name
("thetal_mean", "profiles") => "THETAL",
("temperature_mean", "profiles") => "TABS",
("qt_mean", "profiles") => "QT",
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

                # Calculated and Fixed Variables
                # -- Need to calculate and add ("zc", "profiles"), and ("zc", "reference"), grid shouldn't have to be the same as TC.jl and it should interpolate using vertical_interpolation() from HelperFuncs.jl, if so we need to add z=0 to z/zf and also something at the surface for variables defined on z/zf, the latter part being the hard part lol so I hope we don't have to do that...
                zf_data = new_data.group["profiles"]["zf"][:]
                zc_data = (zf_data[1:end-1] .+ zf_data[2:end]) ./ 2.0
                _new_dims  =  collect((x=="z" ? "zc" : x for x in NC.dimnames(truth_data["z"]))) # replace any "z" in dimnames with "zc"
                NC.defVar(new_data.group["profiles"], "zc", zc_data, (_new_dims); attrib =  truth_data["z"].attrib )
                NC.defVar(new_data.group["reference"], "zc", zc_data, _new_dims; attrib =  truth_data["z"].attrib )
                # -- Need to fix time variable to go from days to second
                new_data.group["profiles"]["t"][:] = (new_data.group["profiles"]["t"][:] .- new_data.group["profiles"]["t"][1]) .* (24.0 .* 3600.0) |> (x -> x .+ (ceil(x[end]/3600)*3600 - x[end])) # shift day to second, then i think t=0 is missing and we need to shift it to end at t=12,14, otherwise would end at 11.91,13.91h so shift to nearest hour
                new_data.group["timeseries"]["t"][:] = (new_data.group["timeseries"]["t"][:] .- new_data.group["timeseries"]["t"][1]) .* (24.0 .* 3600.0 ) |> (x -> x .+ (ceil(x[end]/3600)*3600 - x[end])) # shift day to second, then i think t=0 is missing and we need to shift it to end at t=12,14, otherwise would end at 11.91,13.91h so shift to nearest hour
                #close file at end
                close(new_data)
            else
                @info("Skiping missing file for flight $flight_number and forcing $forcing_type")
            end
        end
    end
end

