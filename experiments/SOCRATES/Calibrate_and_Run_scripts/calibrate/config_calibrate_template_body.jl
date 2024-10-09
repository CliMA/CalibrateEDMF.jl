#= Custom calibration configuration file. =#


# ========================================================================================================================= #
#  This is a template, you should use it in another script in which you set the following items and then include this file  #
# ========================================================================================================================= #

# ========================================================================================================================= #


if calibrate_to == "Atlas_LES"
    @info("Calibrating to Atlas_LES")
    truth_dir = joinpath(main_experiment_dir, "Reference", "Atlas_LES") # the folder where we store our truth (Atlas LES Data)
elseif calibrate_to == "Flight_Observations"
    @info("Calibrating to Flight_Observations")
    truth_dir = joinpath(main_experiment_dir, "Reference", "Flight_Observations", "Faked_Profiles_to_Time") # the folder where we store our truth (Atlas LES Data)
    # based on calibration_vars we need to drop empty slices so the covariance matrix can be calculated without nans so it can still have eigenvalues (we actually need more than one value left over in each row , and in each pair of rows for cov to work)
else
    error("invalid calibrate_to: ", calibrate_to)
end

# Training setups
Train_flight_numbers = flight_numbers
Train_forcing_types = forcing_types
# Validation setups
Val_flight_numbers = flight_numbers
Val_forcing_types = forcing_types

flight_string  = flight_numbers == [1,9,10,11,12,13] ? "All" : join(string.(flight_numbers, pad=2), "_")
forcing_types_str = join((x->split(string(x),"_")[1]).(forcing_types), "_")

# join our defaults to the rest of the namelist

calibration_vars_str = join(sort(calibration_vars), "__")


# ========================================================================================================================= #

@info("supersat_type", supersat_type)
@info("calibration_setup", calibration_setup)
@info("calibration_vars_str", calibration_vars_str)
@info("calibrate_to", calibrate_to)
@info("flight_numbers", flight_numbers)
@info("forcing_types", forcing_types)

# ========================================================================================================================= #

default_user_args = (;
    use_supersat=supersat_type,
    τ_use=:morrison_milbrandt_2015_style,
    use_sedimentation = true, 
    grid_mean_sedimentation = false,
    sedimentation_integration_method = :upwinding,
    use_heterogeneous_ice_nucleation = false,
    sedimentation_ice_number_concentration = supersat_type, # idk if this is good or bad lol...
    liq_velo_scheme = :Chen2022Vel,
    ice_velo_scheme = :Chen2022Vel,
    rain_velo_scheme = :Chen2022Vel,
    snow_velo_scheme = :Chen2022Vel,)

added_user_args::Bool = false
print(added_user_args)
@info("added_user_args", added_user_args)
for (i_i, item) in enumerate(local_namelist)
    if item[1] == "user_args"
        @info("user_args found in local_namelist, merging with default values")
        local_namelist[i_i] = (item[1], merge(default_user_args, item[2])) # i think 2nd means local will overwite default
        global added_user_args = true        
    end
end
@info("added_user_args after is ", added_user_args)
if !added_user_args
    @info("`user_args` not found in local_namelist, adding some default values now")
    local_namelist = [local_namelist; ("user_args", default_user_args)] # shouldn't break anything, add to end (can't use push! bc of type assumptions)
 end

if header_setup_choice == :default
    local_namelist = [default_namelist_args; local_namelist] # add the default namelist args to the local namelist
elseif header_setup_choice == :simple
    local_namelist = [simple_namelist_args; local_namelist] # add the default namelist args to the local namelist
else
    error("invalid header_setup_choice: ", header_setup_choice)
end
@info("local_namelist:", local_namelist)

# ========================================================================================================================= #


# join calibration_parameter dictionaries
if header_setup_choice == :default
    calibration_parameters = merge(default_calibration_parameters, calibration_parameters) # merge the default calibration parameters with the ones we've defined here (overwrite defaults if necessary)
elseif header_setup_choice == :simple
    calibration_parameters = merge(simple_calibration_parameters, calibration_parameters) # merge the default calibration parameters with the ones we've defined here (overwrite defaults if necessary)
else
    error("invalid header_setup_choice: ", header_setup_choice)
end

# consider adding a flag to just get the paths without overwriting existing files
reference_paths = Dict() # for regular atlas_les, keep empty, we'll just generate them later (switch to using the ones generated above?)
include(joinpath(main_experiment_dir, "process_SOCRATES_reference.jl")) # process the truth data into a format we can use for calibration
if calibrate_to == "Flight_Observations" 
    if !ismissing(t_bnds.obs_data)
        @info("Creating trimmed observational obs_data-forced data for calibration")
        _, obs_paths = process_SOCRATES_Flight_Observations_reference(out_vars=calibration_vars, fake_time_bnds = t_bnds.obs_data, out_dir=joinpath(experiment_dir, "Reference", "Flight_Observations"), truth_dir=truth_dir, overwrite=false) # create trimmed data for calibration so the covarainces aren't NaN (failed anyway cause i think after interpolation NaNs can return because the non-NaN data don't form a contiguous rectangle :/ )
    end
    if !ismissing(t_bnds.ERA5_data)
        @info("Creating trimmed observational ERA5_data-forced data for calibration")
        _, ERA5_paths = process_SOCRATES_Flight_Observations_reference(out_vars=calibration_vars, fake_time_bnds = t_bnds.ERA5_data, out_dir=joinpath(experiment_dir, "Reference", "Flight_Observations"), truth_dir=truth_dir, overwrite=false) # create trimmed data for calibration so the covarainces aren't NaN (failed anyway cause i think after interpolation NaNs can return because the non-NaN data don't form a contiguous rectangle :/ )
    end
    # concatenate
    reference_paths = [obs_paths; ERA5_paths]
else
    reference_paths = process_SOCRATES_Atlas_LES_reference(;out_dir=truth_dir, truth_dir=truth_dir, overwrite=false)# the folder where we store our truth (Atlas LES Data) # create trimmed data for calibration so the covarainces aren't NaN (failed anyway cause i think after interpolation NaNs can return because the non-NaN data don't form a contiguous rectangle :/ )
    # reference_paths = Dict() # for regular atlas_les, keep empty, we'll just generate them later (switch to using the ones generated above?)
    # is it time to switch this to just use reference_paths? i don't see why not, what is truth_dir doing otherwise?
end

@info("REFERENCE PATHS: ", reference_paths)


function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(SOCRATES_Train())
    # Define reference used for validation (setting this makes it run validation jobs though..., just use config["reference"] for reruns... and in tc_runner set run_set to reference)
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
    config["outdir_root"] = joinpath(experiment_dir, "Calibrate_and_Run", calibration_setup, calibration_vars_str, "calibrate", "output", calibrate_to, "RF"*flight_string*"_"*forcing_types_str) # store them in the experiment folder here by default, organized by calibration vars
    config["use_outdir_root_as_outdir_path"] = true # use the outdir_root as the directory itself, otherwise it'll create a new directory inside the outdir_root with the calibration parameters
    return config
end

function get_regularization_config()
    config = Dict()
    # Regularization of observations: mean and covariance
    config["perform_PCA"] = perform_PCA # Performs PCA on data
    config["variance_loss"] = variance_loss # Variance truncation level in PCA
    config["normalize"] = true  # whether to normalize data by pooled variance
    config["normalization_type"] = @isdefined(normalization_type) ? normalization_type : :pooled_variance
    config["tikhonov_mode"] = "relative" # Tikhonov regularization
    config["tikhonov_noise"] = 1.0e-6 # Tikhonov regularization
    config["dim_scaling"] = true # Dimensional scaling of the loss

    if normalization_type == :pooled_nonzero_mean_to_value && !@isdefined(obs_var_scaling) # if we didn't set this before, use these defaults
        obs_var_scaling = Dict(
            "temperature_mean" => (1/273.0)^2, #  scale down bc we scaled T to 1, so ΔT is now ∼ 1/273, so scale that up, leave others the same. then ΔT will be O(1) just like Δq
            "ql_mean" => 1.0,
            "qi_mean" => (1.0/5)^2) # scale down so ice becomes more important (factor of 5 rn), maybe will help calibrations...
    end

    config["obs_var_scaling"] = @isdefined(obs_var_scaling) ? obs_var_scaling : nothing # Scale the observation variance by these values

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

    # config["l2_reg"] = Dict(k=>[v["l2_reg"]] for (k,v) in calibration_parameters) # costa said we don't need this, if you leave as nothingw/ augmented true tho it adds cov_prior as regularizer
    # config["l2_reg"] = 0.0 # test no regularization by prior ( i think default is cov(prior))

    # For obs_var_additional_uncertainty_factor, use variable if it exists else nothing
    # config["obs_var_additional_uncertainty_factor"] = Dict(v=>obs_var_additional_uncertainty_factor for v in calibration_vars) # testing.... (separate for each variable)
    config["obs_var_additional_uncertainty_factor"] = obs_var_additional_uncertainty_factor # testing.... (overall), default is nothing unless it was overwritten somewhere

    config["additive_inflation"] = additive_inflation # Additive inflation factor for the covariance matrix (ollie said try starting around here)
    # -- in principle we could put this in model_error as structural error right? but that gets added which isn't exactly what we want...
    
    return config
end


function get_process_config()
    config = Dict()
    config["N_iter"] = N_iter # reduced temporarily for testing
    config["N_ens"] = N_ens # Must be 2p+1 when algorithm is "Unscented"
    config["algorithm"] = "Inversion" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = Δt # EKI learning rate # (maybe change this if having problems?), but the higher above 1 we get the more we forget the prior so.
    @info("Δt: ", config["Δt"])
    @info("alt_scheduler", alt_scheduler)
    if @isdefined(alt_scheduler)
        if alt_scheduler == :default || isnothing(alt_scheduler)
            config["scheduler"] = nothing
            @info("herererererere")
        else
            config["scheduler"] = alt_scheduler
        end
    else
        config["scheduler"] = DataMisfitController(on_terminate = "continue") # costa said this should work better, see , ollie said 'try terminate_at' = some largish number... (however it goes very slowly sometimes lol)
    end
    # config["scheduler"] = DataMisfitController(on_terminate = "continue") # costa said this should work better, see , ollie said 'try terminate_at' = some largish number... (however it goes very slowly sometimes lol)
    config["accelerator"] = DefaultAccelerator()
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = false # costa set this to false? (TESTING FALSE!!!)
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    # use localizer when number of parameters > number of ensemble members
    # config["localizer"] = SEC(0.5, 0.1) # First arg is strength of localization, second is the minimum correlation retained
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
        name = "RF"*string(setup["flight_number"],pad=2)*"_"*string(setup["forcing_type"])
        if (setup["flight_number"], setup["forcing_type"]) in keys(reference_paths)
            datafile = reference_paths[(setup["flight_number"], setup["forcing_type"])] # use the trimmed data for calibration
        else
            datafile = joinpath(truth_dir, name, "stats", name*".nc") # this would be unneeded if we just used refrerence_paths for everything
        end
       if !isnothing(datafile) && isfile(datafile) # this might not work for obs since we do have truth there existing but can't run socrates...
           setup["datafile"] = datafile
           setup["case_name"] = "SOCRATES_"*name
       else
           @warn("File $datafile does not exist")
       end
    end
    setups = filter(d->haskey(d,"datafile"), setups) # remove setups that didn't have a forcing datafile (namely 11 obs)

    # maybe add a filter based on if not truth_dir,  but atlasles dir has a forcing file there there since that's the real limitation, whether youre comparing to atlas les or obs you can't run TC w/o forcing data...
    setups = [setup for setup in setups if  !(setup["flight_number"] == 11 && setup["forcing_type"] == :obs_data)] # remove 11 obs explicitly (testing)

    # filter out files where our trimming led to no data

    NC.Dataset(joinpath(main_experiment_dir, "Reference", "SOCRATES_summary.nc"),"r") do SOCRATES_summary
        for setup in setups # set up the periods we take our means over to match atlas (no idea what to do about the covariances, maybe just take the same values?)
            if setup["forcing_type"] == :obs_data # From Atlas paper, hour 10-12 are used for comparing obs
                setup["t_start"] = ismissing(t_bnds.obs_data) ? 10 * 3600. : t_bnds.obs_data[1] # if we're using the shorter time bounds, use those instead
                setup["t_end"]   = ismissing(t_bnds.obs_data) ? 12 * 3600. : t_bnds.obs_data[2] # if we're using the shorter time bounds, use those instead
            elseif setup["forcing_type"] == :ERA5_data # Use the start and end times from Table 2 in atlas, stored in SOCRATES_summary.nc that we created w/ a Python Jupyter notebook
                _sum = NC.@select(SOCRATES_summary, flight_number == $setup["flight_number"])
                t_start, t_end = _sum["time_bnds"]
                t_ref =  _sum["reference_time"][1] # we know this is hour 12
                t_start, t_end = map( x-> x.value, Dates.Second.([t_start,t_end] .- t_ref)) .+ (12 * 3600) # get the difference in seconds between t_start,t_end and t_ref = 12 hours, and add to the 12 hours to get the final values in seconds
                setup["t_start"] = ismissing(t_bnds.ERA5_data) ? t_start : t_bnds.ERA5_data[1] # if we're using the shorter time bounds, use those instead
                setup["t_end"]   = ismissing(t_bnds.ERA5_data) ? t_end   : t_bnds.ERA5_data[2] # if we're using the shorter time bounds, use those instead
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
    config["y_names"] = repeat([calibration_vars], n_repeat) # the variable we want to calibrate on
    config["y_dir"] = ref_dirs

    # # Use full  timeseries for covariance (for us we just use what)? # for covariance
    config["t_start"] = [setup["t_start"] for setup in setups] # I think should be 10 hr for Obs, and from Table 2 Atlas for ERA5
    config["t_end"] = [setup["t_end"] for setup in setups] # I think should be 12 hr for Obs, and from Table 2 Atlas for ERA5
    config["Σ_t_start"] = [setup["t_start"] for setup in setups] #  use hours 11-13 for comparison
    config["Σ_t_end"] = [setup["t_end"] for setup in setups]  #  use hours 11-13 for comparison
    # if we used our own t_end, use those  ( this doesn't work though for atlas les....)
    if calibrate_to == "Atlas_LES" # then the LES start/end time don't change
        config["time_shift"] = [setup["forcing_type"] == :obs_data ? 12 * 3600. : 14 * 3600. for setup in setups] # The shift is essentially how far back from the end in LES data does the TC data start. Here they start at the same place so it's the full length of the ATLAS LES model (12 for obs, 14 for era), must also be float type
    else # calibrating to obs, we need to match the time_bnds we set for those observations...
        config["time_shift"] = [ismissing(t_bnds[setup["forcing_type"]]) ? (setup["forcing_type"] == :obs_data ? 12 * 3600. : 14 * 3600.) : t_bnds[setup["forcing_type"]] for setup in setups] # The shift is essentially how far back from the end in LES data does the TC data start. Here they start at the same place so it's the full length of the ATLAS LES model (12 for obs, 14 for era), must also be float type
    end
    
    # config["batch_size"] = n_repeat # has to be some divisor of n_repeat, default is n_repeat == length(ref_dirs) == number of setups
    @info(calibration_vars)
    @info("local_namelist: ", local_namelist)
    config["namelist_args"] = repeat([local_namelist],n_repeat) # list of tuples with specific namelist_args, separate from and superior to those from the global ones we use in get_scm_config())
    config["write_full_stats"] = false

    # config["reference_mean"] = nothing # provide a list of one per case, so they can be concatented block-diagonally in ReferenceStats.jl, should be one vector/profile per y_name (i.e. calibration_var)
    # config["reference_cov"] = nothing # provide a list of one per case, so they can be concatented block-diagonally in ReferenceStats.jl, should be one cov matrix per y_name (i.e. calibration_var)

    # testing
    config["z_rectifiers"] = [z_bounds[setup["forcing_type"]][setup["flight_number"]] for setup in setups] # z_bounds is a dictionary of dictionaries, so we index it with the flight number and forcing type to get the z_bounds for that setup

    @info(config["z_rectifiers"])
    @info(typeof(config["z_rectifiers"]))

    config["characteristic_values"] = calibration_vars_characteristic_values
    
    return config
end

function get_reference_config(::SOCRATES_Val)
    # config = Dict()
    # error("Validation has not been implemented yet")
    config = get_reference_config(SOCRATES_Train()) # just reutrn the training config for now...
    return config
end

function get_prior_config()
    # Don't forget to also update these in L2-reg (should we add a warning to check they all have the same keys or something?)
    config = Dict()
    # constraints on each variable
    config["constraints"] = Dict(k=>[wrap(v["constraints"])...] for (k,v) in calibration_parameters) # costa said we don't need this
    # TC.jl prior mean
    config["prior_mean"] = Dict(k=>[wrap(v["prior_mean"])...] for (k,v) in calibration_parameters) # put in list if scalar (expand lists so doesnt become nested)
    # not sure yet what this is lol
    # config["unconstrained_σ"] = 1.0 # just leave everyting variance 
    # Tight initial prior for Unscented
    # config["unconstrained_σ"] = 0.25
    # Test defaulting to 1 but allowing for different values
    config["unconstrained_σ"] = Dict(k=>[wrap(get(v,"unconstrained_σ", 1.0))...] for (k,v) in calibration_parameters)

    # Allow passing in scalars for vector parameter variances..., just gets repeated
    n_repeats = Dict(k=> length(v["prior_mean"]) for (k,v) in calibration_parameters)
    for (k,v) in config["unconstrained_σ"]
        if length(v) != n_repeats[k]
            config["unconstrained_σ"][k] = repeat(config["unconstrained_σ"][k], n_repeats[k] ÷ length(v) )
        end
    end

    return config
end

function get_scm_config() # set all my namelist stuff here, these are global settings.
    config = Dict()
    config["namelist_args"] = [ # this could move out of body if necessary...
        ("time_stepping", "dt_min", 0.5),
        ("time_stepping", "dt_max", 2.0),
        ("time_stepping", "t_max", t_max), # shorter for testing
        ("stats_io", "frequency", 600.0), # long runs so try a lower output rate for smaller files... (seems to be seconds) -- changed to 10 minutes... 14 hours default runs are loooong...
    ]
    return config
end