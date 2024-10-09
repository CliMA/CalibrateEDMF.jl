#= Custom calibration configuration file. =#

# ========================================================================================================================= #
# # this_dir = @__DIR__ # the location of this file (doens't work cause this gets copied and moved around both in HPC and in calibration pipeline)
using CalibrateEDMF
pkg_dir = pkgdir(CalibrateEDMF)
main_experiment_dir = joinpath(pkg_dir, "experiments", "SOCRATES")
include(joinpath(main_experiment_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_header.jl")) # load the template config file for the rest of operations
# ========================================================================================================================= #
# experiment (should match this directory names)
supersat_type = :exponential_T_scaling_ice
# calibration_setup (should match the directory names)
calibration_setup = "tau_autoconv_noneq"
# calibrate_to
calibrate_to = "Atlas_LES"
# SOCRATES setups
flight_numbers = Vector{Int}([1, 9, 10, 11, 12, 13])
# pad flight number to two digit string if is 1 digit
forcing_types = Vector{Symbol}([:obs_data])

experiment_dir = joinpath(pkg_dir, "experiments", "SOCRATES", "subexperiments", "SOCRATES_"*string(supersat_type))
# ========================================================================================================================= #
# tmax
# t_max = 14*3600.0 # 14 hours
# t_bnds = (;obs_data = missing, ERA5_data = missing) # normal full length (I think we need this for full 14 hours bc reference is only 10-12 on obs and variable on ERA5 so we can't use full t_bnds...)
t_max = 7*3600.0 # shorter for testing (remember to change t_start, t_end, Σ_t_start, Σ_t_end in get_reference_config()# shorter for testing
t_bnds = (;obs_data = (t_max-2*3600.    , t_max), ERA5_data = (t_max-2*3600.     , t_max)) # shorter for testing
t_bnds = (;obs_data = (t_max-2*3600.    , t_max), ERA5_data = (t_max-2*3600.     , t_max)) # shorter for testing
# ========================================================================================================================= #
# constants we use here

# ========================================================================================================================= #
# setup stuff (needs to be here for ekp_par_calibration.sbatch to read), can read fine as long as is after "=" sign
N_ens = 100
N_iter = 20
# ========================================================================================================================= #

calibration_parameters = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    ) # these aren't in the default_namelist so where should I put them?

moisture_model = Dict("pow_icenuc_autoconv_eq" => "equilibrium", "tau_autoconv_noneq" => "nonequilibrium")
local_namelist = [ # things in namelist that otherwise wouldn't be... (both random parameters and parameters we want to calibrate that we added ourselves that generate_namelist doesn't insert...)
    #
    ("thermodynamics", "moisture_model", moisture_model[calibration_setup]), # choosing noneq for training...
    #
    # for user_args use the defaults in footer by default
]

include(joinpath(main_experiment_dir, "subexperiments", "SOCRATES_"*string(supersat_type), "Calibrate_and_Run", calibration_setup, "calibration_parameters_and_namelist.jl")) # Load the files
calibration_parameters = merge(calibration_parameters, calibration_parameters__experiment_setup)
local_namelist = vcat(local_namelist, local_namelist__experiment_setup) # add the experiment setup to the local namelist

# ========================================================================================================================= #
calibration_vars = ["ql_mean", "qi_mean", "qr_mean", "qip_mean"]
# ========================================================================================================================= #
# ========================================================================================================================= #

# header_setup_choice = :simple # switch from :default to :simple calibration
header_setup_choice = :default # switch from :default to :simple calibration

# alt_scheduler = :default # missing for use default, nothing for default eki timestepper w/ learning rate
alt_scheduler = DataMisfitController(on_terminate = "continue") # or just don't define it...

variance_loss = 1.0e-5
normalization_type = :pooled_nonzero_mean_to_value # let all variables have nonzero values mean 1, since we don't have a well defined variance or something to look at, and this way we can change our additional uncertainty factor freely

# ========================================================================================================================= #
# ========================================================================================================================= #


include(joinpath(main_experiment_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_footer.jl")) # load the template config file for the rest of operations#= Custom calibration configuration file. =#














