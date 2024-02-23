#= Custom calibration configuration file. =#

# ========================================================================================================================= #
# this_dir = @__DIR__ # the location of this file (doens't work cause this gets copied and moved around both in HPC and in calibration pipeline)
using CalibrateEDMF
pkg_dir = pkgdir(CalibrateEDMF)
main_experiment_dir = joinpath(pkg_dir, "experiments", "SOCRATES")
include(joinpath(main_experiment_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_header.jl")) # load the template config file for the rest of operations
# ========================================================================================================================= #
# experiment (should match this directory names)
supersat_type = :linear_combination_with_w
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
# t_bnds = (;obs_data = missing, ERA5_data = missing) # normal full length
t_max = 2*3600.0 # shorter for testing (remember to change t_start, t_end, Σ_t_start, Σ_t_end in get_reference_config()# shorter for testing
t_bnds = (;obs_data = (0.0    , t_max), ERA5_data = (0.0    , t_max)) # shorter for testing
# ========================================================================================================================= #
# constants we use here
ρ_l = FT(1000.) # density of ice, default from ClimaParameters
ρ_i = FT(916.7) # density of ice, default from ClimaParameters
r_r = FT(20 * 1e-6) # 20 microns
r_0 = FT(.2 * 1e-6) # .2 micron base aerosol

N_l = FT(1e-5 / (4/3 * π * r_r^3 * ρ_l)) # estimated total N assuming reasonable q_liq.. (N = N_r in homogenous)
N_i = FT(1e-7 / (4/3 * π * r_r^3 * ρ_i)) # estimated total N assuming reasonable q_ice... (N = N_r + N_0)

D_ref = FT(0.0000226)
D = D_ref
# ========================================================================================================================= #
# setup stuff (needs to be here for ekp_par_calibration.sbatch to read), can read fine as long as is after "=" sign
N_ens  = 30 # number of ensemble members
N_iter = 20 # number of iterations
# ========================================================================================================================= #
calibration_parameters_default = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    #
    "linear_combination_liq_c_1"   => Dict("prior_mean" => FT(N_l * r_0)                    , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing), # I think at q=0, we need c_1 from linear = c_1 from geometric...
    "linear_combination_liq_c_2"   => Dict("prior_mean" => FT(2/3.)                         , "constraints" => bounded(1/3., 1) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing), # Halfway between 1/3 and 1 (we know these can't be right?) but it has the same sign lmao so it still decays... (we would need to figure out how to match slopes at some arbitrary point near 0 that isn't 0 lmao)
    "linear_combination_liq_c_3"   => Dict("prior_mean" => FT(0)                            , "constraints" => bounded_above(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing), # asssume nothing here? (keep 0 as upper bound?) 
    "linear_combination_liq_c_4"   => Dict("prior_mean" => FT(0)                            , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing), # w up should lead to τ down, so coefficient is positive
    # "linear_combination_liq_c_5"   => Dict("prior_mean" => FT(N_l * r_0)                    , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing), 
    #
    "linear_combination_ice_c_1"   => Dict("prior_mean" => FT(N_i * r_0)                    , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing), # I think at q=0, we need c_1 from linear = c_1 from geometric...
    "linear_combination_ice_c_2"   => Dict("prior_mean" => FT(2/3.)                         , "constraints" => bounded(1/3., 1) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing), # Halfway between 1/3 and 1 (we know these can't be right?) but it has the same sign lmao so it still decays... (we would need to figure out how to match slopes at some arbitrary point near 0 that isn't 0 lmao)
    "linear_combination_ice_c_3"   => Dict("prior_mean" => FT(-0.6)                         , "constraints" => bounded_above(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing), # Fletcher 1962 (values taken from Frostenberg 2022), same sign again I suppose...
    "linear_combination_ice_c_4"   => Dict("prior_mean" => FT(0)                            , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing), #  w up should lead to τ down, so coefficient is positive
    # "linear_combination_ice_c_5"   => Dict("prior_mean" => FT(N_l * r_0)                    , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing), 
    #
    "τ_acnv_rai"      => Dict("prior_mean" => FT(default_params["τ_acnv_rai"]["value"])      , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "rain_autoconversion_timescale"), # bounded_below(0) = bounded(0,Inf) from EnsembleKalmanProcesses.jl
    "τ_acnv_sno"      => Dict("prior_mean" => FT(default_params["τ_acnv_sno"]["value"])      , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "snow_autoconversion_timescale"), 
    "q_liq_threshold" => Dict("prior_mean" => FT(default_params["q_liq_threshold"]["value"]) , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "cloud_liquid_water_specific_humidity_autoconversion_threshold"),
    "q_ice_threshold" => Dict("prior_mean" => FT(default_params["q_ice_threshold"]["value"]) , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "cloud_ice_specific_humidity_autoconversion_threshold"),
    ) # these aren't in the default_namelist so where should I put them?
calibration_parameters = deepcopy(calibration_parameters_default) # copy the default parameters and edit them below should we ever wish to change this

# global local_namelist = [] # i think if you use something like local_namelist = ... below inside the function it will just create a new local variable and not change this one, so we need to use global (i think we didnt need after switching to local_namelist_here but idk...)
local_namelist = [ # things in namelist that otherwise wouldn't be... (both random parameters and parameters we want to calibrate that we added ourselves that generate_namelist doesn't insert...)
    ("user_aux", "linear_combination_liq_c_1", calibration_parameters["linear_combination_liq_c_1"]["prior_mean"] ),
    ("user_aux", "linear_combination_liq_c_2", calibration_parameters["linear_combination_liq_c_2"]["prior_mean"] ),
    ("user_aux", "linear_combination_liq_c_3", calibration_parameters["linear_combination_liq_c_3"]["prior_mean"] ),
    ("user_aux", "linear_combination_liq_c_4", calibration_parameters["linear_combination_liq_c_4"]["prior_mean"] ),
    # ("user_aux", "linear_combination_liq_c_5", calibration_parameters["linear_combination_liq_c_5"]["prior_mean"] ),
    #
    ("user_aux", "linear_combination_ice_c_1", calibration_parameters["linear_combination_ice_c_1"]["prior_mean"] ),
    ("user_aux", "linear_combination_ice_c_2", calibration_parameters["linear_combination_ice_c_2"]["prior_mean"] ),
    ("user_aux", "linear_combination_ice_c_3", calibration_parameters["linear_combination_ice_c_3"]["prior_mean"] ),
    ("user_aux", "linear_combination_ice_c_4", calibration_parameters["linear_combination_ice_c_4"]["prior_mean"] ),
    # ("user_aux", "linear_combination_ice_c_5", calibration_parameters["linear_combination_ice_c_5"]["prior_mean"] ),
    #
    ("user_aux", "min_τ_liq", FT( 3.)), # stability testing
    ("user_aux", "min_τ_ice", FT( 3.)), # stability testing
    #
    ("thermodynamics", "moisture_model", "nonequilibrium"), # choosing noneq for training...
    ("thermodynamics", "sgs", "mean"), # sgs has to be mean in noneq
    ("user_args", (;use_supersat=supersat_type) ) # we need supersat for non_eq results and the ramp for eq
]
@info("local_namelist:", local_namelist)

calibration_vars = ["temperature_mean", "ql_mean","qi_mean"]
# calibration_vars = ["temperature_mean", "qt_mean"] # qt instead of ql or qi because we don't care about phase just yet, temperature_mean to make sure other stuff doesn't get out of hand.

# ========================================================================================================================= #
# ========================================================================================================================= #

include(joinpath(main_experiment_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_body.jl")) # load the template config file for the rest of operations