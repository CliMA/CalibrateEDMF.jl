#= Custom calibration configuration file. =#

# ========================================================================================================================= #
# this_dir = @__DIR__ # the location of this file (doens't work cause this gets copied and moved around both in HPC and in calibration pipeline)
using CalibrateEDMF
pkg_dir = pkgdir(CalibrateEDMF)
main_experiment_dir = joinpath(pkg_dir, "experiments", "SOCRATES")
include(joinpath(main_experiment_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_header.jl")) # load the template config file for the rest of operations
# ========================================================================================================================= #
# experiment (should match this directory names)
supersat_type = :Base
# calibration_setup (should match the directory names)
calibration_setup = "tau_autoconv_noneq"
# calibrate_to
calibrate_to = "Atlas_LES" # "Atlas_LES" or "Flight_Observations"
# SOCRATES setups
flight_numbers = [1,9,10,11,12,13]
# pad flight number to two digit string if is 1 digit
forcing_types  = [:obs_data]

experiment_dir = joinpath(pkg_dir, "experiments", "SOCRATES", "subexperiments", "SOCRATES_"*string(supersat_type))
# ========================================================================================================================= #
# tmax
# t_max = 14*3600.0 # 14 hours
# t_bnds = (;obs_data = missing, ERA5_data = missing) # normal full length (I think we need this for full 14 hours bc reference is only 10-12 on obs and variable on ERA5 so we can't use full t_bnds...)
t_max = 7*3600.0 # shorter for testing (remember to change t_start, t_end, Σ_t_start, Σ_t_end in get_reference_config()# shorter for testing
t_bnds = (;obs_data = (0.0    , t_max), ERA5_data = (0.0    , t_max)) # shorter for testing
t_bnds = (;obs_data = (t_max-2*3600.    , t_max), ERA5_data = (t_max-2*3600.     , t_max)) # shorter for testing

# ========================================================================================================================= #
# constants we use here

# ========================================================================================================================= #
# setup stuff (needs to be here for ekp_par_calibration.sbatch to read), can read fine as long as is after "=" sign, only one of these lines each can exist bc they're grepped for in the sbatch script...
N_ens  = 100 # number of ensemble members (neede)
N_iter = 15 # number of iterations
# ========================================================================================================================= #
expanded_unconstrained_σ = FT(2.0) # alternate for unconstrained_σ when we truly don't know the prior mean (or it's very uncertain)  ## TESTING 100 HERE!!!! (seem to have some nan errors...) (bounded_below values are log spaced  so consider that)
calibration_parameters_default = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    #
    # "τ_sub_dep"      => Dict("prior_mean" => FT(default_params["τ_sub_dep"]["value"])        , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "sublimation_deposition_timescale"), # bounded_below(0) = bounded(0,Inf) from EnsembleKalmanProcesses.jl
    "τ_sub_dep"      => Dict("prior_mean" => FT(1000.)        , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "sublimation_deposition_timescale", "unconstrained_σ" => expanded_unconstrained_σ), # starting up here bc I think it's more stable...
    "τ_cond_evap"    => Dict("prior_mean" => FT(default_params["τ_cond_evap"]["value"])      , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "condensation_evaporation_timescale", "unconstrained_σ" => expanded_unconstrained_σ), 
) # these aren't in the default_namelist so where should I put them?
calibration_parameters = deepcopy(calibration_parameters_default) # copy the default parameters and edit them below should we ever wish to change this

# global local_namelist = [] # i think if you use something like local_namelist = ... below inside the function it will just create a new local variable and not change this one, so we need to use global (i think we didnt need after switching to local_namelist_here but idk...)
local_namelist = [ # things in namelist that otherwise wouldn't be... (both random parameters and parameters we want to calibrate that we added ourselves that generate_namelist doesn't insert...)
    #
    ("microphysics", "τ_sub_dep"  , calibration_parameters["τ_sub_dep"  ]["prior_mean"] ),
    ("microphysics", "τ_cond_evap", calibration_parameters["τ_cond_evap"]["prior_mean"] ),
    #
    # ("user_aux", "min_τ_liq", FT( 3.)), # stability testing
    # ("user_aux", "min_τ_ice", FT( 3.)), # stability testing
    #
    ("thermodynamics", "moisture_model", "nonequilibrium"), # choosing noneq for training...
    ("thermodynamics", "sgs", "mean"), # sgs has to be mean in noneq
    # ("user_args", (;use_supersat=supersat_type) ) # we need supersat for non_eq results and the ramp for eq
    # ("user_args", (;use_supersat=supersat_type, τ_use=:morrison_milbrandt_2015_style_exponential_part_only) ), # we need supersat for non_eq results and the ramp for eq, testing
    ("user_args", (;use_supersat=supersat_type, τ_use=:morrison_milbrandt_2015_style) ), # testing if this improves stability...
    #
    ("turbulence", "EDMF_PrognosticTKE", "max_area", FT(.3)), # stability limiting...
]
@info("local_namelist:", local_namelist)

calibration_vars = ["temperature_mean", "ql_mean","qi_mean"]
# calibration_vars = ["temperature_mean", "qt_mean"] # qt instead of ql or qi because we don't care about phase just yet, temperature_mean to make sure other stuff doesn't get out of hand.
# calibration_vars = ["ql_mean","qi_mean"]

# ========================================================================================================================= #
# ========================================================================================================================= #


obs_var_additional_uncertainty_factor = 0.3 # I hope this is in scaled space lmao... (so we'd be adding a variance of 1.0*obs_var value to the observation error variance), hope the mean is order 1 so the variance is reasonable...

header_setup_choice = :simple # switch from :default to :simple calibration

alt_scheduler = :default # missing for use default, nothing for default eki timestepper w/ learning rate
# ========================================================================================================================= #
# ========================================================================================================================= #

include(joinpath(main_experiment_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_body.jl")) # load the template config file for the rest of operations#= Custom calibration configuration file. =#



