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
FT = Float64

# Cases defined as structs for quick access to default configs
struct SOCRATES_Train end
struct SOCRATES_Val end

# this_dir = @__DIR__ # the location of this file (doens't work cause this gets copied and moved around both in HPC and in calibration pipeline)
supersat_type = :exponential_T_scaling_ice
pkg_dir = pkgdir(CalibrateEDMF)
experiment_dir = joinpath(pkg_dir, "experiments", "SOCRATES_" * string(supersat_type))

calibrate_to = "Atlas_LES" # "Atlas_LES" or "Flight_Observations"
if calibrate_to == "Atlas_LES"
    @info("Calibrating to Atlas_LES")
    truth_dir = joinpath(experiment_dir, "Reference", "Atlas_LES") # the folder where we store our truth (Atlas LES Data)
elseif calibrate_to == "Flight_Observations"
    @info("Calibrating to Flight_Observations")
    truth_dir = joinpath(experiment_dir, "Reference", "Flight_Observations", "Faked_Profiles_to_Time") # the folder where we store our truth (Atlas LES Data)
# based on calibration_vars we need to drop empty slices so the covariance matrix can be calculated without nans so it can still have eigenvalues (we actually need more than one value left over in each row , and in each pair of rows for cov to work)
else
    error("invalid calibrate_to: ", calibrate_to)
end

# SOCRATES setups
flight_numbers = [1, 9, 10, 11, 12, 13]
forcing_types = [:obs_data]
# Training setups
Train_flight_numbers = flight_numbers
Train_forcing_types = [:obs_data]
# Validation setups
Val_flight_numbers = flight_numbers
Val_forcing_types = forcing_types

# t_max = 14*3600.0 # 14 hours
# t_bnds = (;obs_data = missing, ERA5_data = missing) # normal full length
t_max = 2 * 3600.0 # shorter for testing (remember to change t_start, t_end, Σ_t_start, Σ_t_end in get_reference_config()# shorter for testing
t_bnds = (; obs_data = (0.0, t_max), ERA5_data = (0.0, t_max)) # shorter for testing


default_params = CalibrateEDMF.HelperFuncs.CP.create_toml_dict(FT; dict_type = "alias") # name since we use the alias in this package, get a list of all default_params (including those not in driver/generate_namelist.jl) from ClimaParameters
calibration_parameters_default = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    "τ_cond_evap" => Dict(
        "prior_mean" => FT(default_params["τ_cond_evap"]["value"]),
        "constraints" => bounded_below(0),
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => "condensation_evaporation_timescale",
    ),
    #
    "T_scaling_c_1" => Dict(
        "prior_mean" => FT(0.02),
        "constraints" => bounded_below(0),
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => nothing,
    ),
    "T_scaling_c_2" => Dict(
        "prior_mean" => FT(-0.6),
        "constraints" => bounded_above(0),
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => nothing,
    ),
    #
    "τ_acnv_rai" => Dict(
        "prior_mean" => FT(default_params["τ_acnv_rai"]["value"]),
        "constraints" => bounded_below(0),
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => "rain_autoconversion_timescale",
    ), # bounded_below(0) = bounded(0,Inf) from EnsembleKalmanProcesses.jl
    "τ_acnv_sno" => Dict(
        "prior_mean" => FT(default_params["τ_acnv_sno"]["value"]),
        "constraints" => bounded_below(0),
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => "snow_autoconversion_timescale",
    ),
    "q_liq_threshold" => Dict(
        "prior_mean" => FT(default_params["q_liq_threshold"]["value"]),
        "constraints" => bounded_below(0),
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => "cloud_liquid_water_specific_humidity_autoconversion_threshold",
    ),
    "q_ice_threshold" => Dict(
        "prior_mean" => FT(default_params["q_ice_threshold"]["value"]),
        "constraints" => bounded_below(0),
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => "cloud_ice_specific_humidity_autoconversion_threshold",
    ),
) # these aren't in the default_namelist so where should I put them?
calibration_parameters = deepcopy(calibration_parameters_default) # copy the default parameters and edit them below should we ever wish to change this

global local_namelist = [] # i think if you use something like local_namelist = ... below inside the function it will just create a new local variable and not change this one, so we need to use global
local_namelist = [
    ("microphysics", "τ_cond_evap", calibration_parameters["τ_cond_evap"]["prior_mean"]),
    #
    ("user_aux", "T_scaling_c_1", calibration_parameters["T_scaling_c_1"]["prior_mean"]),
    ("user_aux", "T_scaling_c_2", calibration_parameters["T_scaling_c_2"]["prior_mean"]),
]
@info("local_namelist:", local_namelist)

calibration_vars = ["temperature_mean", "ql_mean", "qi_mean"]
# calibration_vars = ["temperature_mean", "qt_mean"] # qt instead of ql or qi because we don't care about phase just yet, temperature_mean to make sure other stuff doesn't get out of hand.

# process_truth # I dont think we need this anymore, just change NaNs in the covariance to 0s and use nancov()
# consider adding a flag to just get the paths without overwriting existing files
include(joinpath(experiment_dir, "process_SOCRATES_reference.jl")) # process the truth data into a format we can use for calibration
if calibrate_to == "Flight_Observations"
    if !ismissing(t_bnds.obs_data)
        @info("Creating trimmed observational obs_data-forced data for calibration")
        _, obs_paths = process_SOCRATES_Flight_Observations_reference(
            out_vars = calibration_vars,
            fake_time_bnds = t_bnds.obs_data,
            out_dir = joinpath(experiment_dir, "Reference", "Flight_Observations"),
            overwrite = false,
        ) # create trimmed data for calibration so the covarainces aren't NaN (failed anyway cause i think after interpolation NaNs can return because the non-NaN data don't form a contiguous rectangle :/ )
    end
    if !ismissing(t_bnds.ERA5_data)
        @info("Creating trimmed observational ERA5_data-forced data for calibration")
        _, ERA5_paths = process_SOCRATES_Flight_Observations_reference(
            out_vars = calibration_vars,
            fake_time_bnds = t_bnds.ERA5_data,
            out_dir = joinpath(experiment_dir, "Reference", "Flight_Observations"),
            overwrite = false,
        ) # create trimmed data for calibration so the covarainces aren't NaN (failed anyway cause i think after interpolation NaNs can return because the non-NaN data don't form a contiguous rectangle :/ )
    end
    # concatenate
    reference_paths = [obs_paths; ERA5_paths]
else
    process_SOCRATES_Atlas_LES_reference(; out_dir = truth_dir, overwrite = false)# the folder where we store our truth (Atlas LES Data) # create trimmed data for calibration so the covarainces aren't NaN (failed anyway cause i think after interpolation NaNs can return because the non-NaN data don't form a contiguous rectangle :/ )
    reference_paths = Dict() # for regular atlas_les, keep empty, we'll just generate them later (switch to using the ones generated above?
end

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
    config["outdir_root"] = joinpath(
        experiment_dir,
        "Calibrate_and_Run",
        "tau_autoconv_noneq",
        "calibrate",
        "output",
        calibrate_to,
        "RFAll_obs",
    ) # store them in the experiment folder here by default
    config["use_outdir_root_as_outdir_path"] = true # use the outdir_root as the directory itself, otherwise it'll create a new directory inside the outdir_root with the calibration parameters
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

    # config["l2_reg"] = Dict(k=>[v["l2_reg"]] for (k,v) in calibration_parameters) # costa said we don't need this
    return config
end


function get_process_config()
    config = Dict()
    config["N_iter"] = 10 # reduced temporarily for testing
    config["N_ens"] = 15 # Must be 2p+1 when algorithm is "Unscented"
    config["algorithm"] = "Inversion" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = 1.0 # EKI learning rate
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = true
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    return config
end

function get_reference_config(::SOCRATES_Train)
    config = Dict()

    # Setup SOCRATES run arguments I guess
    flight_numbers = Train_flight_numbers
    forcing_types = Train_forcing_types
    aux_kwargs = () # fill in later

    setups = collect(Iterators.product(flight_numbers, forcing_types))[:]
    setups = map(x -> Dict("flight_number" => x[1], "forcing_type" => x[2]), setups) # convert to list of dictionaries
    # add this here because it seems to be called?
    for setup in setups
        name = "RF" * string(setup["flight_number"], pad = 2) * "_" * string(setup["forcing_type"])
        if (setup["flight_number"], setup["forcing_type"]) in keys(reference_paths)
            setup["datafile"] = reference_paths[(setup["flight_number"], setup["forcing_type"])] # use the trimmed data for calibration
        else
            datafile = joinpath(truth_dir, name, "stats", name * ".nc")
        end
        if !isnothing(datafile) && isfile(datafile) # this might not work for obs since we do have truth there existing but can't run socrates...
            setup["datafile"] = datafile
            setup["case_name"] = "SOCRATES_" * name
        else
            @warn("File $datafile does not exist")
        end
    end
    setups = filter(d -> haskey(d, "datafile"), setups) # remove setups that didn't have a forcing datafile (namely 11 obs)

    # maybe add a filter based on if not truth_dir,  but atlasles dir has a forcing file there there since that's the real limitation, whether youre comparing to atlas les or obs you can't run TC w/o forcing data...
    setups = [setup for setup in setups if !(setup["flight_number"] == 11 && setup["forcing_type"] == :obs_data)] # remove 11 obs explicitly (testing)

    # filter out files where our trimming led to no data

    NC.Dataset(joinpath(experiment_dir, "Reference", "SOCRATES_summary.nc"), "r") do SOCRATES_summary
        for setup in setups # set up the periods we take our means over to match atlas (no idea what to do about the covariances, maybe just take the same values?)
            if setup["forcing_type"] == :obs_data # From Atlas paper, hour 10-12 are used for comparing obs
                setup["t_start"] = ismissing(t_bnds.obs_data) ? 10 * 3600.0 : t_bnds.obs_data[1] # if we're using the shorter time bounds, use those instead
                setup["t_end"] = ismissing(t_bnds.obs_data) ? 12 * 3600.0 : t_bnds.obs_data[2] # if we're using the shorter time bounds, use those instead
            elseif setup["forcing_type"] == :ERA5_data # Use the start and end times from Table 2 in atlas, stored in SOCRATES_summary.nc that we created w/ a Python Jupyter notebook
                _sum = NC.@select(SOCRATES_summary, flight_number == $setup["flight_number"])
                t_start, t_end = _sum["time_bnds"]
                t_ref = _sum["reference_time"][1] # we know this is hour 12
                t_start, t_end = map(x -> x.value, Dates.Second.([t_start, t_end] .- t_ref)) .+ (12 * 3600) # get the difference in seconds between t_start,t_end and t_ref = 12 hours, and add to the 12 hours to get the final values in seconds
                setup["t_start"] = ismissing(t_bnds.ERA5_data) ? t_start : t_bnds.ERA5_data[1] # if we're using the shorter time bounds, use those instead
                setup["t_end"] = ismissing(t_bnds.ERA5_data) ? t_end : t_bnds.ERA5_data[2] # if we're using the shorter time bounds, use those instead
            else
                error("invalid forcing_type: ", setup["forcing_type"])
            end
        end
    end # SOCRATES_summary is closed

    setups = filter(d -> haskey(d, "datafile"), setups) # filter out if datafile doesn't exist (11 :obsdata for example)
    n_repeat = length(setups)
    ref_dirs = [dirname(setup["datafile"]) for setup in setups]

    # need reference dirs from wherever i put my truth, maybe add a SOCRATESUtils to match les_utils etc.
    n_repeat = length(ref_dirs)
    config["case_name"] = [setup["case_name"] for setup in setups]
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = LES() # our reference is atlas LES
    config["Σ_reference_type"] = LES()
    config["y_names"] = repeat([calibration_vars], n_repeat) # the variable we want to calibrate on
    config["y_dir"] = ref_dirs

    # # Use full  timeseries for covariance (for us we just use what)? # for covariance
    config["t_start"] = [setup["t_start"] for setup in setups] # I think should be 10 hr for Obs, and from Table 2 Atlas for ERA5
    config["t_end"] = [setup["t_end"] for setup in setups] # I think should be 12 hr for Obs, and from Table 2 Atlas for ERA5
    config["Σ_t_start"] = [setup["t_start"] for setup in setups] #  use hours 11-13 for comparison
    config["Σ_t_end"] = [setup["t_end"] for setup in setups]  #  use hours 11-13 for comparison
    # if we used our own t_end, use those  ( this doesn't work though for atlas les....)
    if calibrate_to == "Atlas_LES" # then the LES start/end time don't change
        config["time_shift"] = [setup["forcing_type"] == :obs_data ? 12 * 3600.0 : 14 * 3600.0 for setup in setups] # The shift is essentially how far back from the end in LES data does the TC data start. Here they start at the same place so it's the full length of the ATLAS LES model (12 for obs, 14 for era), must also be float type
    else # calibrating to obs, we need to match the time_bnds we set for those observations...
        config["time_shift"] = [
            ismissing(t_bnds[setup["forcing_type"]]) ?
            (setup["forcing_type"] == :obs_data ? 12 * 3600.0 : 14 * 3600.0) : t_bnds[setup["forcing_type"]] for
            setup in setups
        ] # The shift is essentially how far back from the end in LES data does the TC data start. Here they start at the same place so it's the full length of the ATLAS LES model (12 for obs, 14 for era), must also be float type
    end

    # config["batch_size"] = n_repeat # has to be some divisor of n_repeat, default is n_repeat == length(ref_dirs) == number of setups
    @info(calibration_vars)
    @info(local_namelist)
    local_namelist_here = [
        ("thermodynamics", "moisture_model", "nonequilibrium"), # choosing noneq for training...
        ("thermodynamics", "sgs", "mean"), # sgs has to be mean in noneq
        ("user_args", (; use_supersat = supersat_type)), # we need supersat for non_eq results and the ramp for eq
    ]
    local_namelist_here = [local_namelist; local_namelist_here] # overwrite_namelist | # list of tuples (<namelist_section>, <namelist_key>, <value>) matching namelist[<namelist_section>][<namelist_key>] = <value>, don't append cause it'll keep growing lol...
    @info("local_namelist_here: ", local_namelist_here)
    config["namelist_args"] = repeat([local_namelist_here], n_repeat) # list of tuples with specific namelist_args, separate from and superior to those from the global ones we use in get_scm_config())
    config["write_full_stats"] = false

    # config["reference_mean"] = nothing # provide a list of one per case, so they can be concatented block-diagonally in ReferenceStats.jl, should be one vector/profile per y_name (i.e. calibration_var)
    # config["reference_cov"] = nothing # provide a list of one per case, so they can be concatented block-diagonally in ReferenceStats.jl, should be one cov matrix per y_name (i.e. calibration_var)

    return config
end

function get_reference_config(::SOCRATES_Val)
    config = Dict()
    error("Validation has not been implemented yet")
    return config
end

function get_prior_config()
    # Don't forget to also update these in L2-reg (should we add a warning to check they all have the same keys or something?)
    config = Dict()
    # constraints on each variable
    config["constraints"] = Dict(k => [v["constraints"]] for (k, v) in calibration_parameters) # costa said we don't need this
    # TC.jl prior mean
    config["prior_mean"] = Dict(k => [v["prior_mean"]] for (k, v) in calibration_parameters) # costa said we don't need this
    # not sure yet what this is lol
    config["unconstrained_σ"] = 1.0 # just leave everyting variance 
    # Tight initial prior for Unscented
    # config["unconstrained_σ"] = 0.25
    return config
end

function get_scm_config() # set all my namelist stuff here, these are global settings.
    config = Dict()
    config["namelist_args"] = [
        ("time_stepping", "dt_min", 0.5),
        ("time_stepping", "dt_max", 2.0),
        ("time_stepping", "t_max", t_max), # shorter for testing
        ("stats_io", "frequency", 600.0), # long runs so try a lower output rate for smaller files... (seems to be seconds) -- changed to 10 minutes... 14 hours default runs are loooong...
    ]
    return config
end
