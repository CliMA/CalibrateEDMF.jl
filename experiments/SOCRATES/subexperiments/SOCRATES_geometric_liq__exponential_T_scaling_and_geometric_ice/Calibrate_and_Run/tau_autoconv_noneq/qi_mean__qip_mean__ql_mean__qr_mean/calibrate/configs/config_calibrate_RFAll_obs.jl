#= Custom calibration configuration file. =#

# ========================================================================================================================= #
# this_dir = @__DIR__ # the location of this file (doens't work cause this gets copied and moved around both in HPC and in calibration pipeline)
using CalibrateEDMF
pkg_dir = pkgdir(CalibrateEDMF)
main_experiment_dir = joinpath(pkg_dir, "experiments", "SOCRATES")
include(joinpath(main_experiment_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_header.jl")) # load the template config file for the rest of operations
# ========================================================================================================================= #
# experiment (should match this directory names)
supersat_type = :geometric_liq__exponential_T_scaling_and_geometric_ice
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
t_max = 7*3600.0 # shorter for testing (remember to change t_start, t_end, Σ_t_start, Σ_t_end in get_reference_config()# shorter for testing
# t_bnds = (;obs_data = (0.0    , t_max), ERA5_data = (0.0    , t_max)) # shorter for testing
t_bnds = (;obs_data = (t_max-2*3600.    , t_max), ERA5_data = (t_max-2*3600.     , t_max)) # shorter for testing

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
N_ens  = 100 # number of ensemble members
N_iter = 15 # number of iterations
# ========================================================================================================================= #

bounded_σ = FT(2.0) # most of the log space is pressed up against the edges so not too big, assume it's closer to linear than log
bounded_edge_σ = FT(5.0) # is in log space go crazy (10 orders of magnitude is pretty big though)
unbounded_σ = FT(5.0) # is in real space so... good luck lol

calibration_parameters_default = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    #
    "geometric_liq_c_1"   => Dict("prior_mean" => FT(1/(4/3 * π * ρ_l * r_r^2))    , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ), # Inhomogenous default assuming r_0 = 20 micron since `typical` N is harder to define
    "geometric_liq_c_2"   => Dict("prior_mean" => FT(2/3.)                         , "constraints" => bounded(1/3., 1) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_σ), # Halfway between 1/3 and 1
    "geometric_liq_c_3"   => Dict("prior_mean" => FT(N_l * r_0)                    , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ), 
    #
    "geometric_and_T_scaling_ice_c_1"   => Dict("prior_mean" => FT((4*π*D) * ((4/3 * π * ρ_i)^(-1/3)  * (N_i)^(2/3) * (0.02)^(2/3) + (N_i * r_0))) , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ), # Yeahhh.... idk for this one lol... just combined them serially from the homogenous case where c_3 is -1/3
    "geometric_and_T_scaling_ice_c_2"   => Dict("prior_mean" => FT(2/3.)                                                                           , "constraints" => bounded(1/3., 1) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_σ),# Halfway between 1/3 and 1 -- should this be the same as c_2g? It's the same mixing...
    "geometric_and_T_scaling_ice_c_3"   => Dict("prior_mean" => FT((4*π*D) * r_0 * .02 )                                                           , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ), # Fletcher 1962 (values taken from Frostenberg 2022)
    "geometric_and_T_scaling_ice_c_4"   => Dict("prior_mean" => FT(-0.6)                                                                           , "constraints" => bounded_above(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ), # Fletcher 1962 (values taken from Frostenberg 2022)
    #
    "r_ice_snow" => Dict("prior_mean" => FT(62.5e-6)    , "constraints" => bounded(0, 1e-3) , "l2_reg" => nothing, "CLIMAParameters_longname" => "r_ice_snow", "unconstrained_σ" => bounded_σ),
    "ice_sedimentation_scaling_factor" => Dict("prior_mean" => FT(1.0) , "constraints" => bounded(0,5) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_σ), # testing
) # these aren't in the default_namelist so where should I put them?
calibration_parameters = deepcopy(calibration_parameters_default) # copy the default parameters and edit them below should we ever wish to change this

# global local_namelist = [] # i think if you use something like local_namelist = ... below inside the function it will just create a new local variable and not change this one, so we need to use global (i think we didnt need after switching to local_namelist_here but idk...)
local_namelist = [ # things in namelist that otherwise wouldn't be... (both random parameters and parameters we want to calibrate that we added ourselves that generate_namelist doesn't insert...)
    ("user_aux", "geometric_liq_c_1", calibration_parameters["geometric_liq_c_1"]["prior_mean"] ),
    ("user_aux", "geometric_liq_c_2", calibration_parameters["geometric_liq_c_2"]["prior_mean"] ),
    ("user_aux", "geometric_liq_c_3", calibration_parameters["geometric_liq_c_3"]["prior_mean"] ),
    #
    ("user_aux", "geometric_and_T_scaling_ice_c_1", calibration_parameters["geometric_and_T_scaling_ice_c_1"]["prior_mean"] ),
    ("user_aux", "geometric_and_T_scaling_ice_c_2", calibration_parameters["geometric_and_T_scaling_ice_c_2"]["prior_mean"] ),
    ("user_aux", "geometric_and_T_scaling_ice_c_3", calibration_parameters["geometric_and_T_scaling_ice_c_3"]["prior_mean"] ),
    ("user_aux", "geometric_and_T_scaling_ice_c_4", calibration_parameters["geometric_and_T_scaling_ice_c_4"]["prior_mean"] ),
    #
    # ("user_aux", "min_τ_liq", FT( 3.)), # stability testing
    # ("user_aux", "min_τ_ice", FT( 3.)), # stability testing
    #
    ("user_aux", "adjust_ice_N", true),
    ("microphysics", "r_ice_snow", calibration_parameters["r_ice_snow"]["prior_mean"] ),
    ("user_aux", "ice_sedimentation_scaling_factor", calibration_parameters["ice_sedimentation_scaling_factor"]["prior_mean"] ),
    ("user_aux", "ice_sedimentation_Dmax", Inf), # 62.5 microns cutoff from CM
    # ("user_aux", "τ_ice_scaling_factor", calibration_parameters["τ_ice_scaling_factor"]["prior_mean"] ),
    #
    ("thermodynamics", "moisture_model", "nonequilibrium"), # choosing noneq for training...
    ("thermodynamics", "sgs", "mean"), # sgs has to be mean in noneq

    # ("user_args", (;use_supersat=supersat_type) ), # we need supersat for non_eq results and the ramp for eq
    # ("user_args", (;use_supersat=supersat_type, τ_use=:morrison_milbrandt_2015_style_exponential_part_only) ), # we need supersat for non_eq results and the ramp for eq, testing
    ("user_args", (;
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
        snow_velo_scheme = :Chen2022Vel,)),
    #
    ("turbulence", "EDMF_PrognosticTKE", "max_area", FT(.3)), # stability limiting...
]
# namelist["user_aux"]["sedimentation_Dmax"] = 62.5e-6 # 62.5 microns cutoff from CM

@info("local_namelist:", local_namelist)

calibration_vars = ["ql_mean", "qi_mean", "qr_mean", "qip_mean"]

# ========================================================================================================================= #
# ========================================================================================================================= #

obs_var_additional_uncertainty_factor = 0.1 # I hope this is in scaled space lmao... (so we'd be adding a variance of 1.0*obs_var value to the observation error variance), hope the mean is order 1 so the variance is reasonable...
obs_var_additional_uncertainty_factor = Dict( # I hope this is in scaled space lmao... (so we'd be adding a variance of 1.0*obs_var value to the observation error variance), hope the mean is order 1 so the variance is reasonable...
    "temperature_mean" => obs_var_additional_uncertainty_factor / 273, # scale down bc it's already so big for temperature, so divide by characteristic value to get ΔT ∼ obs_var_additional_uncertainty_factor instead... more like additive lol (note it's in normalized space but still)
    "ql_mean"          => obs_var_additional_uncertainty_factor,
    "qi_mean"          => obs_var_additional_uncertainty_factor,
)

# header_setup_choice = :simple # switch from :default to :simple calibration
header_setup_choice = :default # switch from :default to :simple calibration

alt_scheduler = DataMisfitController(on_terminate = "continue") # or just don't define it...

# alt_scheduler = :default # missing for use default, nothing for default eki timestepper w/ learning rate
# Δt = 8.0 # testing a larger one... (can also retry dmc since failures are reduced)

normalization_type = :pooled_nonzero_mean_to_value # let all variables have nonzero values mean 1, since we don't have a well defined variance or something to look at, and this way we can change our additional uncertainty factor freely

# ========================================================================================================================= #
# ========================================================================================================================= #


include(joinpath(main_experiment_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_body.jl")) # load the template config file for the rest of operations