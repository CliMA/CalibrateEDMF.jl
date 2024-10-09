
# ========================================================================================================================= #
#  This is a template, you should use it in another script in which you set the following items and then include this file  #
# ========================================================================================================================= #

using Distributions
using StatsBase
using LinearAlgebra
using Random
# using CalibrateEDMF
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

default_params = CalibrateEDMF.HelperFuncs.CP.create_toml_dict(FT; dict_type="alias") # name since we use the alias in this package, get a list of all default_params (including those not in driver/generate_namelist.jl) from ClimaParameters

wrap(x) = (x isa Union{Vector,Tuple}) ? x : [x] # wrap scalars in a vector so that they can be splatted regardless of what they are (then you can pass in eitehr vector or scalar in your calibration_parameters dict)




# ========================================================================================================================= #
# ========================================================================================================================= #

@warn("Add method to save other variables to the diagnostics file, maybe something similar to get_profile, but for different y_names..., and only in real space")
@warn("rn run_SCM calls eval_single_ref_model() which calls get_profile() so we just need to add another call to get out")
@warn("they use NetCDFIO_Diags so we need to make up something similar somehow...")

# ========================================================================================================================= #
# ========================================================================================================================= #

NUM_NN_PARAMS = 12 # number of neural network parameters from Costa
linear_entr_unc_sigma = repeat([5.0], Int(NUM_NN_PARAMS / 2))
linear_detr_unc_sigma = repeat([5.0], Int(NUM_NN_PARAMS / 2))
linear_entr_unc_sigma[end] = 0.5
linear_detr_unc_sigma[end] = 0.5
non_vec_sigma = 2.0 # costa's σ for non-vector parameters

# entrainment setup things from costa
linear_entr_prior_mean = 0.5 .* (rand(Int(NUM_NN_PARAMS / 2)) .- 0.5) .+ 1.0
linear_detr_prior_mean = 0.5 .* (rand(Int(NUM_NN_PARAMS / 2)) .- 0.5) .+ 1.0
linear_entr_prior_mean[end] = 0.01
linear_detr_prior_mean[end] = 0.01
# linear_entr_prior_mean[Int(NUM_NN_PARAMS / 2)] = 1.0
# linear_detr_prior_mean[end] = 1.0


# ========================================================================================================================= #
# Global Default Params
# ========================================================================================================================= #
global_param_defaults = Dict()
global_param_defaults["τ_cond_evap"] = Dict("prior_μ" => FT(100.), "σ" => 2.5) # a little higher so we can spread it out more relative to ice and allow it to survive even w/ precip


r_r = FT(20 * 1e-6) # 20 microns
ρ_l = FT(1000.)
q_r_1 = FT((4/3) * π * r_r^3 * 1000) # mass of one raindrop, q = N q_r
B = FT(100) # Hong and Lim 2006 [ Constant in raindrop freezing equation ]
global_param_defaults["heterogeneous_ice_nuclation_coefficient"] = q_r_1/ρ_l * B # See the writeup
global_param_defaults["heterogeneous_ice_nuclation_exponent"] = 0.66 # Hong & Lim 2006 [Constant in Biggs Freezing]

_max_area = FT(0.3)

# ========================================================================================================================= #
# ========================================================================================================================= #

path_to_Costa_SOTA = "/groups/esm/cchristo/cedmf_results/james_v1_runs/results_Inversion_p22_e300_i15_mb_LES_2024-03-15_10-08_Vxo_longer_long_run/Diagnostics.nc"
CEDMF_path = pkgdir(CalibrateEDMF.ModelTypes) # any submodule should work the same
include(joinpath(CEDMF_path, "tools", "DiagnosticsTools.jl")) # provides optimal_parameters()
Costa_SOTA = optimal_parameters(path_to_Costa_SOTA, method = "last_nn_particle_mean")
Costa_SOTA = Dict(zip(Costa_SOTA...)) # turn to dict

Costa_SOTA_linear_ent_params = [Costa_SOTA["linear_ent_params_{$(i)}"] for i in 1:NUM_NN_PARAMS]

Costa_SOTA_namelist_args = [
    # Entrainment Params
    ("turbulence", "EDMF_PrognosticTKE", "linear_ent_params", Costa_SOTA_linear_ent_params),
    # Diffusion Params
    ("turbulence", "EDMF_PrognosticTKE", "tke_ed_coeff", Costa_SOTA["tke_ed_coeff"]),
    ("turbulence", "EDMF_PrognosticTKE", "tke_diss_coeff", Costa_SOTA["tke_diss_coeff"]),
    ("turbulence", "EDMF_PrognosticTKE", "static_stab_coeff", Costa_SOTA["static_stab_coeff"]),
    ("turbulence", "EDMF_PrognosticTKE", "tke_surf_scale", Costa_SOTA["tke_surf_scale"]),
    ("turbulence", "EDMF_PrognosticTKE", "Prandtl_number_0", Costa_SOTA["Prandtl_number_0"]),
    # Momentum Exchange parameters
    ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_adv_coeff", Costa_SOTA["pressure_normalmode_adv_coeff"]),
    ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_buoy_coeff1", Costa_SOTA["pressure_normalmode_buoy_coeff1"]),
    ("turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_drag_coeff", Costa_SOTA["pressure_normalmode_drag_coeff"]),
    # Area Limiters
    ("turbulence", "EDMF_PrognosticTKE", "min_area_limiter_scale", Costa_SOTA["min_area_limiter_scale"]),
    ("turbulence", "EDMF_PrognosticTKE", "min_area_limiter_power", Costa_SOTA["min_area_limiter_power"]),
]
@info("Costa_SOTA_namelist_args", Costa_SOTA_namelist_args)
# ========================================================================================================================= #
# ========================================================================================================================= #

unconstrained_σ_autoconversion_timescale = FT(3.0) # alternate for unconstrained_σ when we truly don't know the prior mean (or it's very uncertain)
unconstrained_σ_autoconversion_threshold = FT(4.0) # alternate for unconstrained_σ when we truly don't know the prior mean (or it's very uncertain) (but this is log space so maybe a bit wider since we're less certain on thresholds)

bounded_edge_σ = FT(3.0) # is in log space go crazy (10 orders of magnitude is pretty big though)
bounded_σ = FT(2.0) # most of the log space is pressed up against the edges so not too big, assume it's closer to linear than log

autoconversion_calibration_parameters = Dict(
    # autoconversion
    "τ_acnv_rai"      => Dict("prior_mean" => FT(1000) , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "rain_autoconversion_timescale", "unconstrained_σ" => unconstrained_σ_autoconversion_timescale), # bounded_below(0) = bounded(0,Inf) from EnsembleKalmanProcesses.jl
    "τ_acnv_sno"      => Dict("prior_mean" => FT(1000) , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => "snow_autoconversion_timescale", "unconstrained_σ" => unconstrained_σ_autoconversion_timescale), 
    "q_liq_threshold" => Dict("prior_mean" => FT(1e-4) , "constraints" => bounded(0, 1e-3) , "l2_reg" => nothing, "CLIMAParameters_longname" => "cloud_liquid_water_specific_humidity_autoconversion_threshold", "unconstrained_σ" => unconstrained_σ_autoconversion_threshold), # unrealistically high upper bound
    "q_ice_threshold" => Dict("prior_mean" => FT(1e-6) , "constraints" => bounded(0, 1e-4) , "l2_reg" => nothing, "CLIMAParameters_longname" => "cloud_ice_specific_humidity_autoconversion_threshold", "unconstrained_σ" => unconstrained_σ_autoconversion_threshold),          # unrealistically high upper bound
)

ice_sedimentation_parameters = Dict(
    "ice_sedimentation_scaling_factor" => Dict("prior_mean" => FT(0.1) , "constraints" => bounded(0,5) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_σ), # testing
    "τ_ice_scaling_factor" => Dict("prior_mean" => FT(1.0) , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => FT(1)), # .1 to 10 should be reasonable lol
)


# Things we're calibrating (from costa are the entrainment/detrainment)
default_calibration_parameters = Dict(
    # entrainment parameters
    # "linear_ent_params" => Dict("prior_mean" => FT.(zeros(NUM_NN_PARAMS)), "constraints" => [repeat([no_constraint()], NUM_NN_PARAMS)...], "l2_reg" => nothing, "CLIMAParameters_longname" => "linear_ent_params", "unconstrained_σ" => cat(linear_entr_unc_sigma, linear_detr_unc_sigma, dims = 1)),

    # diffusion parameters 
    # "tke_ed_coeff" => Dict("prior_mean" => FT(0.14), "constraints" => bounded(0.01, 1.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "tke_ed_coeff", "unconstrained_σ" => non_vec_sigma ),
    # "tke_diss_coeff" => Dict("prior_mean" => FT(0.22), "constraints" => bounded(0.01, 1.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "tke_diss_coeff", "unconstrained_σ" => non_vec_sigma),
    # "static_stab_coeff" => Dict("prior_mean" => FT(0.4), "constraints" => bounded(0.01, 1.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "static_stab_coeff", "unconstrained_σ" => non_vec_sigma),
    # "tke_surf_scale" => Dict("prior_mean" => FT(3.75), "constraints" => bounded(1.0, 16.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "tke_surf_scale", "unconstrained_σ" => non_vec_sigma),
    # "Prandtl_number_0" => Dict("prior_mean" => FT(0.74), "constraints" => bounded(0.5, 1.5), "l2_reg" => nothing, "CLIMAParameters_longname" => "Prandtl_number_0", "unconstrained_σ" => non_vec_sigma),

    # momentum exchange parameters 
    # "pressure_normalmode_adv_coeff" => Dict("prior_mean" => FT(0.001), "constraints" => bounded(0.0, 100.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "pressure_normalmode_adv_coeff", "unconstrained_σ" => non_vec_sigma),
    # "pressure_normalmode_buoy_coeff1" => Dict("prior_mean" => FT(0.12), "constraints" => bounded(0.0, 10.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "pressure_normalmode_buoy_coeff1", "unconstrained_σ" => non_vec_sigma),
    # "pressure_normalmode_drag_coeff" => Dict("prior_mean" => FT(10.0), "constraints" => bounded(0.0, 100.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "pressure_normalmode_drag_coeff", "unconstrained_σ" => non_vec_sigma),

    # # Area limiters (turned off rn bc was tuned to 0.9 and need to figure how how to make it more stable and peak around 0.3 w/ SOCRATES grid spacing)
    "area_limiter_scale" => Dict("prior_mean" => FT(5.0), "constraints" => bounded(0.0, 100.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "area_limiter_scale", "unconstrained_σ" => non_vec_sigma ), # these were set w/ max_area = 0.3 in mind
    "area_limiter_power" => Dict("prior_mean" => FT(30.0), "constraints" => bounded(1.0, 50.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "area_limiter_power", "unconstrained_σ" => non_vec_sigma ),  # these were set w/ max_area = 0.3 in mind
    # "min_area_limiter_scale" => Dict("prior_mean" => FT(2.0), "constraints" => bounded(1.0, 100.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "min_area_limiter_scale", "unconstrained_σ" => 1.0 ),
    # "min_area_limiter_power" => Dict("prior_mean" => FT(2000.0), "constraints" => bounded(1000.0, 5000.0), "l2_reg" => nothing, "CLIMAParameters_longname" => "min_area_limiter_power", "unconstrained_σ" => 1.0 ),
    # surface area (important for updraft evolution? -- should this be bounded by max_area?
    "surface_area" => Dict("prior_mean" => FT(default_params["surface_area"]["value"]), "constraints" => bounded(0.0, _max_area), "l2_reg" => nothing, "CLIMAParameters_longname" => "surface_area", "unconstrained_σ" => non_vec_sigma),
    #
    # "r_ice_snow" => Dict("prior_mean" => FT(62.5e-6)    , "constraints" => bounded(0, 1e-3) , "l2_reg" => nothing, "CLIMAParameters_longname" => "r_ice_snow", "unconstrained_σ" => bounded_σ),
)



default_calibration_parameters = merge(default_calibration_parameters, autoconversion_calibration_parameters, ice_sedimentation_parameters)


# Thing's were setting and not calibrating
default_namelist_args = [ # from Costa, i think just starting here is fine and if we overwrite things later they'll just overwrite in the namelist too?
    # ("time_stepping", "dt_min", 0.5),
    # ("time_stepping", "dt_max", 5.0),
    # ("stats_io", "frequency", 60.0),
    # ("time_stepping", "t_max", 3600.0 * SCM_RUN_TIME_HR),
    # ("t_interval_from_end_s", 3600.0 * SCM_RUN_TIME_HR),
    # ("thermodynamics", "sgs", "quadrature"), # doesnt work w/ noneq
    # ("turbulence", "EDMF_PrognosticTKE", "surface_area_bc", "Prognostic"), # Doesn't work w/ -ΔT from socratessummary
    # ("turbulence", "EDMF_PrognosticTKE", "surface_area_bc", "Fixed"),
    # Add namelist_args defining entrainment closure, e.g.
    ("turbulence", "EDMF_PrognosticTKE", "entrainment_type", "total_rate"),
    ("turbulence", "EDMF_PrognosticTKE", "entr_dim_scale", "w_height"),
    ("turbulence", "EDMF_PrognosticTKE", "detr_dim_scale", "mf_grad"),
    ("turbulence", "EDMF_PrognosticTKE", "turbulent_entrainment_factor", 0.0),
    ("turbulence", "EDMF_PrognosticTKE", "entrainment", "None"),
    ("turbulence", "EDMF_PrognosticTKE", "ml_entrainment", "Linear"),
    ("turbulence", "EDMF_PrognosticTKE", "min_area", 1e-10),
    ("turbulence", "EDMF_PrognosticTKE", "limit_min_area", false),
    ("turbulence", "EDMF_PrognosticTKE", "area_limiter_scale", 0.0), # if having problems with area limiter, try removing this line
    ("turbulence", "EDMF_PrognosticTKE", "entr_pi_subset", (1, 2, 3, 4, 6)),
    # ("turbulence", "EDMF_PrognosticTKE", "pi_norm_consts", [1e6, 1e3, 1.0, 1.0, 1.0, 1.0]),
    ("turbulence", "EDMF_PrognosticTKE", "pi_norm_consts", [100.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
    # ("turbulence", "EDMF_PrognosticTKE", "linear_ent_params", zeros(NUM_NN_PARAMS)),
    ("turbulence", "EDMF_PrognosticTKE", "linear_ent_biases", true),
]


# TEST
default_namelist_args = [ # from Costa, i think just starting here is fine and if we overwrite things later they'll just overwrite in the namelist too?
    # ("time_stepping", "dt_min", 0.5),
    # ("time_stepping", "dt_max", 5.0),
    # ("stats_io", "frequency", 60.0),
    # ("time_stepping", "t_max", 3600.0 * SCM_RUN_TIME_HR),
    # ("t_interval_from_end_s", 3600.0 * SCM_RUN_TIME_HR),
    # ("thermodynamics", "sgs", "quadrature"), # doesnt work w/ noneq
    # ("turbulence", "EDMF_PrognosticTKE", "surface_area_bc", "Prognostic"), # try turning this off (is more stable now, don't think it's causing the lack of convergence problems either...)  # Doesn't work w/ -ΔT from socratessummary
    # ("turbulence", "EDMF_PrognosticTKE", "surface_area_bc", "Fixed"),
    # Add namelist_args defining entrainment closure, e.g.
    ("turbulence", "EDMF_PrognosticTKE", "entrainment_type", "total_rate"),
    ("turbulence", "EDMF_PrognosticTKE", "entr_dim_scale", "w_height"),
    ("turbulence", "EDMF_PrognosticTKE", "detr_dim_scale", "mf_grad"),
    ("turbulence", "EDMF_PrognosticTKE", "turbulent_entrainment_factor", 0.0),
    ("turbulence", "EDMF_PrognosticTKE", "entrainment", "None"),
    ("turbulence", "EDMF_PrognosticTKE", "ml_entrainment", "Linear"),
    ("turbulence", "EDMF_PrognosticTKE", "min_area", 1e-10),
    ("turbulence", "EDMF_PrognosticTKE", "limit_min_area", false),
    # ("turbulence", "EDMF_PrognosticTKE", "area_limiter_scale", 0.0), # if having problems with area limiter, try removing this line
    # ("turbulence", "EDMF_PrognosticTKE", "area_limiter_power", 5.0),
    ("turbulence", "EDMF_PrognosticTKE", "max_area", _max_area), # stability limiting... can overwrite later
    ("turbulence", "EDMF_PrognosticTKE", "entr_pi_subset", (1, 2, 3, 4, 6)),
    # ("turbulence", "EDMF_PrognosticTKE", "pi_norm_consts", [1e6, 1e3, 1.0, 1.0, 1.0, 1.0]),
    ("turbulence", "EDMF_PrognosticTKE", "pi_norm_consts", [100.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
    # ("turbulence", "EDMF_PrognosticTKE", "linear_ent_params", zeros(NUM_NN_PARAMS)),
    ("turbulence", "EDMF_PrognosticTKE", "linear_ent_biases", true),
    #
    ("user_aux", "adjust_ice_N", true),
    # ("microphysics", "r_ice_snow", calibration_parameters["r_ice_snow"]["prior_mean"] ),
    ("user_aux", "ice_sedimentation_scaling_factor", ice_sedimentation_parameters["ice_sedimentation_scaling_factor"]["prior_mean"] ),
    ("user_aux", "τ_ice_scaling_factor", ice_sedimentation_parameters["τ_ice_scaling_factor"]["prior_mean"] ),
    ("user_aux", "ice_sedimentation_Dmax", FT(Inf)), # 62.5 microns cutoff from CM
    
    
]

default_namelist_args = [Costa_SOTA_namelist_args; default_namelist_args] # add the Costa SOTA namelist args to the default ones

# ========================================================================================================================= #
# can we add a default user_args that we add to/merge with later in _body.jl? yes, it's in _body.jl
# ========================================================================================================================= #
# ========================================================================================================================= #


# Has almost no failures :)

simple_namelist_args = [ # from Costa, i think just starting here is fine and if we overwrite things later they'll just overwrite in the namelist too?
    ("turbulence", "EDMF_PrognosticTKE", "max_area", _max_area), # stability limiting... can overwrite later
]

simple_calibration_parameters = autoconversion_calibration_parameters

# ========================================================================================================================= #
# ========================================================================================================================= #

default_namelist_constraints = Dict() # from Costa, to fill in (not sure I used this?)

# obs_var_additional_uncertainty_factor = nothing # testing if adding variance that is a factor times the obs value helps regularization... (for cases where variance is 0...)
obs_var_additional_uncertainty_factor = 0.1 # I hope this is in scaled space lmao... (so we'd be adding a variance of 1.0*obs_var value to the observation error variance), hope the mean is order 1 so the variance is reasonable...
obs_var_additional_uncertainty_factor = Dict( # I hope this is in scaled space lmao... (so we'd be adding a variance of 1.0*obs_var value to the observation error variance), hope the mean is order 1 so the variance is reasonable...
    "temperature_mean" => obs_var_additional_uncertainty_factor / 273, # scale down bc it's already so big for temperature, so divide by characteristic value to get ΔT ∼ obs_var_additional_uncertainty_factor instead... more like additive lol (note it's in normalized space but still)
    "ql_mean"          => obs_var_additional_uncertainty_factor,
    "qi_mean"          => obs_var_additional_uncertainty_factor,
    "qr_mean"          => obs_var_additional_uncertainty_factor,
    "qs_mean"          => obs_var_additional_uncertainty_factor,
    "ql_all_mean"      => obs_var_additional_uncertainty_factor,
    "qi_all_mean"      => obs_var_additional_uncertainty_factor,
    "qt_mean"          => obs_var_additional_uncertainty_factor,

)

# set some characteristic values to serve as default to normalize by if a profile is all 0's. This is hard though bc if youre the wrong order of magnitude compared to the simulation you could swamp your MSE.
calibration_vars_characteristic_values = Dict(
    "temperature_mean" => FT(273), # 273 K
    "ql_mean"          => FT(1e-4), # 1e-3 kg/kg
    "qi_mean"          => FT(1e-8), # 1e-3 kg/kg
    "qr_mean"          => FT(1e-2), # 1e-3 kg/kg
    "qs_mean"          => FT(1e-3), # 1e-3 kg/kg
    "ql_all_mean"      => FT(1e-3), # 1e-3 kg/kg
    "qi_all_mean"      => FT(1e-8), # 1e-3 kg/kg
    "qt_mean"          => FT(1e-3), # 1e-3 kg/kg
)
    
# Can maybe implement this later, rn we're just using obs_var_scaling to adjust
# pooled_nonzero_mean_values = Dict(
#     "ql_mean" => FT(1),
#     "ql_all_mean" => FT(1),
#     #
#     "qi_mean" => FT(5), # 5 times more important
#     "qi_all_mean" => FT(5), # 5 times more important
# )

header_setup_choice = :default # can be overwritten in the exepriment script, currently have choices :default and :simple

@warn("How is additive inflation (at least I think it is) changing the absolute scaling when using obs_var_scaling? do we actually need to add a 'normalize to' value?")
additive_inflation = nothing # default to not using it
Δt = 1.0

perform_PCA = true

variance_loss = 1.0e-2

# default is 1, can implement changing that later w/ pooled_nonzero_mean_values
normalization_type = :pooled_nonzero_mean_to_value # let all variables have nonzero values mean 1, since we don't have a well defined variance or something to look at, and this way we can change our additional uncertainty factor freely

# ========================================================================================================================= #
# ========================================================================================================================= #

#=
As far as I can tell

- [ ] In src/TurbulenceConvectionUtils.j, eval_single_ref_model() get's z_obs using get_z_obs(), and then calls get_profile using that z_obs().... Thus it should be possible to tell the sum to only use z within a range...
- [ ] In src/ReferenceModels.jl, the reference model gets z_obs using construct_z_obs() which calls TurbulenceConvection.construct_mesh() from TurbulenceConvection.jl/driver/common_spaces.jl...However there's no interpolation (obviously)Thus it should be possible to insert an attack here... but it will be a little annoying...



You could 
- [ ] Edit the reference files to match your desired (z_min, z_max), z_profile, etcThen the simulations would interpolate to this just fine in eval_single_ref_model()
- [ ] Edit the get_z_obs call to return the z you want to z_scm, nd trust get_obs() to call get_profile(), interpolate and sort things out for you. But unclear if this propagates properly everywhere...In this version you do NOT have access to config, so you need to add a flag to ReferenceStatistics

ReferenceStatistics() is called in Pipeline.jl so you've gotta make some edits there... (in get_ref_stats_kwargs() that is called inside)
=#

z_bounds = Dict(
    :obs_data => Dict{Int, NTuple{2, Union{FT, Missing}}}(
        1 => (0, 4000),
        9 => (0, 4000),
        10 => (0, 4000),
        11 => (missing, missing),
        12 => (0, 2000),
        13 => (0, 2000), ),
    :ERA5_data => Dict{Int, NTuple{2, Union{FT, Missing}}}(
        1 => (missing, missing),
        9 => (missing, missing),
        10 => (missing, missing),
        11 => (missing, missing),
        12 => (missing, missing),
        )
)
