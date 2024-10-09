#= Custom calibration configuration file. =#

# ========================================================================================================================= #
# this_dir = @__DIR__ # the location of this file (doens't work cause this gets copied and moved around both in HPC and in calibration pipeline)
using CalibrateEDMF
pkg_dir = pkgdir(CalibrateEDMF)
main_experiment_dir = joinpath(pkg_dir, "experiments", "SOCRATES")
include(joinpath(main_experiment_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_header.jl")) # load the template config file for the rest of operations
# ========================================================================================================================= #
# experiment (should match this directory names)
supersat_type = :neural_network
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
ρ_l = 1000. # density of ice, default from ClimaParameters
ρ_i = 916.7 # density of ice, default from ClimaParameters
r_0 = 20 * 1e-6 # 20 microns
# ========================================================================================================================= #
# setup stuff (needs to be here for ekp_par_calibration.sbatch to read), can read fine as long as is after "=" sign
N_ens  = 150 # number of ensemble members (neede)
N_iter = 20 # number of iterations
# ========================================================================================================================= #
# NN stuff
nn_path = joinpath(experiment_dir, "Calibrate_and_Run", calibration_setup, "calibrate", "pretrained_NN.jld2")
nn_pretrained_params, nn_pretrained_repr, nn_pretrained_x_0_characteristic = JLD2.load(nn_path, "params", "re", "x_0_characteristic")
# ========================================================================================================================= #
calibration_parameters_default = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    #
    "neural_microphysics_relaxation_network"   => Dict("prior_mean" => FT.(nn_pretrained_params)  , "constraints" => repeat([no_constraint()], length(nn_pretrained_params)) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing),  # have to use one FT throughout
    #
    "τ_acnv_rai"      => Dict("prior_mean" => FT(default_params["τ_acnv_rai"]["value"])      , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "rain_autoconversion_timescale"), # bounded_below(0) = bounded(0,Inf) from EnsembleKalmanProcesses.jl
    "τ_acnv_sno"      => Dict("prior_mean" => FT(default_params["τ_acnv_sno"]["value"])      , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "snow_autoconversion_timescale"), 
    "q_liq_threshold" => Dict("prior_mean" => FT(default_params["q_liq_threshold"]["value"]) , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "cloud_liquid_water_specific_humidity_autoconversion_threshold"),
    "q_ice_threshold" => Dict("prior_mean" => FT(default_params["q_ice_threshold"]["value"]) , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "cloud_ice_specific_humidity_autoconversion_threshold"),
    ) # these aren't in the default_namelist so where should I put them?
calibration_parameters = deepcopy(calibration_parameters_default) # copy the default parameters and edit them below should we ever wish to change this

# global local_namelist = [] # i think if you use something like local_namelist = ... below inside the function it will just create a new local variable and not change this one, so we need to use global (i think we didnt need after switching to local_namelist_here but idk...)
local_namelist = [ # things in namelist that otherwise wouldn't be...
    ("user_aux", "neural_microphysics_relaxation_network", calibration_parameters["neural_microphysics_relaxation_network"]["prior_mean"]),
    #
    ("user_aux", "model_re_location", nn_path), # i think nn_pretrained_repr is not isbits() so we can't use it in the namelist, so we use the path instead
    ("user_aux", "model_x_0_characteristic", FT.(nn_pretrained_x_0_characteristic)), # have to convert to FT for going in params etc...
    #
    ("user_aux", "min_τ_liq", FT(  3.)), # stability testing
    ("user_aux", "min_τ_ice", FT(  3.)), # stability testing
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