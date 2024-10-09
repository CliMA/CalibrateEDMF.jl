
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

CEDMF_path = pkgdir(CalibrateEDMF.ModelTypes) # any submodule should work the same
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

# entrainment setup things from costa
linear_entr_prior_mean = 0.5 .* (rand(Int(NUM_NN_PARAMS / 2)) .- 0.5) .+ 1.0
linear_detr_prior_mean = 0.5 .* (rand(Int(NUM_NN_PARAMS / 2)) .- 0.5) .+ 1.0
linear_entr_prior_mean[end] = 0.01
linear_detr_prior_mean[end] = 0.01
# linear_entr_prior_mean[Int(NUM_NN_PARAMS / 2)] = 1.0
# linear_detr_prior_mean[end] = 1.0


# ========================================================================================================================= #
# Global Default Params (anything shared across experiments/setups)
# ========================================================================================================================= #

non_vec_sigma = 2.0 # costa's σ for non-vector parameters
expanded_unconstrained_σ = FT(2.5) # alternate for unconstrained_σ when we truly don't know the prior mean (or it's very uncertain)  ## TESTING 100 HERE!!!! (seem to have some nan errors...) (bounded_below values are log spaced  so consider that)

bounded_σ = FT(2.0) # bounded on both sides -- most of the log space is pressed up against the edges so not too big, assume it's closer to linear than log

unbounded_σ = FT(5.0) # is in real space so... good luck lol
expanded_unbounded_σ = FT(50.0) # alternate for unconstrained_σ when we truly don't know the prior mean (or it's very uncertain) 

bounded_edge_σ = FT(3.0) # bounded on one side --  is in log space go crazy (10 orders of magnitude is pretty big though)
expanded_bounded_edge_σ = FT(5.0) # alternate for unconstrained_σ when we truly don't know the prior mean (or it's very uncertain)  ## TESTING 100 HERE!!!! (seem to have some nan errors...) (bounded_below values are log spaced  so consider that)


global_param_defaults = Dict()



r_r = FT(20 * 1e-6) # 20 microns
ρ_l = FT(1000.)

ρ_l = FT(1000.) # density of ice, default from ClimaParameters
ρ_i = FT(916.7) # density of ice, default from ClimaParameters
r_r = FT(20 * 1e-6) # 20 microns
r_0 = FT(.2 * 1e-6) # .2 micron base aerosol

N_0   = FT(100*10^6)
N_l   = FT(1e-5 / (4/3 * π * r_r^3 * ρ_l)) # estimated total N assuming reasonable q_liq.. (N = N_r in homogenous)
N_i   = FT(1e-7 / (4/3 * π * r_r^3 * ρ_i)) # estimated total N assuming reasonable q_ice... (N = N_r + N_0)

D_ref = FT(0.0000226)
D = D_ref

# --------------------------------------------------------------------------------------------- #
using NCDatasets
fit_parameters = NCDatasets.Dataset(joinpath(CEDMF_path, "experiments", "SOCRATES", "Reference", "Output_Inferred_Data", "SOCRATES_Atlas_LES_inferred_parameters.nc"), "r")
use_fit_parameters::Bool = true
priors = Dict{String, FT}(
    "geometric_liq_c_1" => use_fit_parameters ? FT(fit_parameters["geometric_liq_c_1"][1]) : FT(N_l * r_0),
    "geometric_liq_c_2" => use_fit_parameters ? FT(fit_parameters["geometric_liq_c_2"][1]) : FT(1/3),
    "geometric_liq_c_3" => use_fit_parameters ? FT(fit_parameters["geometric_liq_c_3"][1]) : FT(N_l * r_0),

    "geometric_ice_c_1" => use_fit_parameters ? FT(fit_parameters["geometric_ice_c_1"][1]) : FT(1/(4/3 * π * ρ_i * r_r^2)),
    "geometric_ice_c_2" => use_fit_parameters ? FT(fit_parameters["geometric_ice_c_2"][1]) : FT(2/3),
    "geometric_ice_c_3" => use_fit_parameters ? FT(fit_parameters["geometric_ice_c_3"][1]) : FT(N_i * r_0),

    "exponential_T_scaling_ice_c_1" => use_fit_parameters ? FT(fit_parameters["exponential_T_scaling_ice_c_1"][1]) : FT(0.02),
    "exponential_T_scaling_ice_c_2" => use_fit_parameters ? FT(fit_parameters["exponential_T_scaling_ice_c_2"][1]) : FT(-0.6),

    "powerlaw_T_scaling_ice_c_1" => use_fit_parameters ? FT(fit_parameters["powerlaw_T_scaling_ice_c_1"][1]) : FT(-9),
    "powerlaw_T_scaling_ice_c_2" => use_fit_parameters ? FT(fit_parameters["powerlaw_T_scaling_ice_c_2"][1]) : FT(9),

    "exponential_T_scaling_and_geometric_ice_c_1" => use_fit_parameters ? FT(fit_parameters["exponential_T_scaling_and_geometric_ice_c_1"][1]) : FT((4*π*D) * ((4/3 * π * ρ_i)^(-1/3)  * (N_i)^(2/3) * (0.02)^(2/3) + (N_i * r_0))),
    "exponential_T_scaling_and_geometric_ice_c_2" => use_fit_parameters ? FT(fit_parameters["exponential_T_scaling_and_geometric_ice_c_2"][1]) : FT(2/3),
    "exponential_T_scaling_and_geometric_ice_c_3" => use_fit_parameters ? FT(fit_parameters["exponential_T_scaling_and_geometric_ice_c_3"][1]) : FT((4*π*D) * r_0 * .02),
    "exponential_T_scaling_and_geometric_ice_c_4" => use_fit_parameters ? FT(fit_parameters["exponential_T_scaling_and_geometric_ice_c_4"][1]) : FT(-0.6),

    "linear_combination_liq_c_1" => use_fit_parameters ? FT(fit_parameters["linear_combination_liq_c_1"][1]) : FT(N_l * r_0),
    "linear_combination_liq_c_2" => use_fit_parameters ? FT(fit_parameters["linear_combination_liq_c_2"][1]) : FT(0),
    "linear_combination_liq_c_3" => use_fit_parameters ? FT(fit_parameters["linear_combination_liq_c_3"][1]) : FT(1),
    "linear_combination_liq_c_4" => use_fit_parameters ? FT(fit_parameters["linear_combination_liq_c_4"][1]) : FT(1),

    "linear_combination_ice_c_1" => use_fit_parameters ? FT(fit_parameters["linear_combination_ice_c_1"][1]) : FT(N_i * r_0),
    "linear_combination_ice_c_2" => use_fit_parameters ? FT(fit_parameters["linear_combination_ice_c_2"][1]) : FT(-0.6),
    "linear_combination_ice_c_3" => use_fit_parameters ? FT(fit_parameters["linear_combination_ice_c_3"][1]) : FT(1),
    "linear_combination_ice_c_4" => use_fit_parameters ? FT(fit_parameters["linear_combination_ice_c_4"][1]) : FT(1),
)

constraints = Dict{String, Any}( # is this necessary? should be the same perhaps either way? idk...
    "geometric_liq_c_1" => use_fit_parameters ? bounded_below(0) : bounded_below(0),
    "geometric_liq_c_2" => use_fit_parameters ? bounded(1/3, 1) : bounded(1/3, 1),
    "geometric_liq_c_3" => use_fit_parameters ? bounded_below(0) : bounded_below(0),

    "geometric_ice_c_1" => use_fit_parameters ? bounded_below(0) : bounded_below(0),
    "geometric_ice_c_2" => use_fit_parameters ? bounded(1/3, 1) : bounded(1/3, 1),
    "geometric_ice_c_3" => use_fit_parameters ? bounded_below(0) : bounded_below(0),

    "exponential_T_scaling_ice_c_1" => use_fit_parameters ? bounded_below(0) : bounded_below(0), # why do these not match?
    "exponential_T_scaling_ice_c_2" => use_fit_parameters ? bounded_above(0) : bounded_above(0),

    "powerlaw_T_scaling_ice_c_1" => use_fit_parameters ? no_constraint() : bounded_below(0),
    "powerlaw_T_scaling_ice_c_2" => use_fit_parameters ? bounded_below(0) : bounded_below(0),

    "exponential_T_scaling_and_geometric_ice_c_1" => use_fit_parameters ? bounded_below(0) : bounded_below(0),
    "exponential_T_scaling_and_geometric_ice_c_2" => use_fit_parameters ? bounded(1/3, 10) : bounded(1/3, 1), # ours doesnt come out in 1/3 1 by default, is more like 6... not sure how to combine processes... but figure i'll give it a chance
    "exponential_T_scaling_and_geometric_ice_c_3" => use_fit_parameters ? bounded_below(0) : bounded_below(0),
    "exponential_T_scaling_and_geometric_ice_c_4" => use_fit_parameters ? bounded_above(0) : bounded_above(0),

    "linear_combination_liq_c_1" => use_fit_parameters ? no_constraint() : no_constraint(),
    "linear_combination_liq_c_2" => use_fit_parameters ? no_constraint() : no_constraint(), # is this right for constraint? or is it actually unbounded
    "linear_combination_liq_c_3" => use_fit_parameters ? no_constraint() : no_constraint(), # T shouldn't be a factor per se, can go either way
    "linear_combination_liq_c_4" => use_fit_parameters ? no_constraint() : no_constraint(), # w no idea

    "linear_combination_ice_c_1" => use_fit_parameters ? no_constraint() : no_constraint(),
    "linear_combination_ice_c_2" => use_fit_parameters ? bounded_above(0) : bounded_above(0), # enforce temperature direction
    "linear_combination_ice_c_3" => use_fit_parameters ? no_constraint() : no_constraint(), # let q scaling go either way, sometimes it favors fewer larger droplets as q rises for example...
    "linear_combination_ice_c_4" => use_fit_parameters ? no_constraint() : no_constraint(), # w no idea
)

unconstrained_σ = Dict{String, FT}(
    "geometric_liq_c_1" => use_fit_parameters ? bounded_edge_σ : bounded_edge_σ,
    "geometric_liq_c_2" => use_fit_parameters ? bounded_σ : bounded_σ,
    "geometric_liq_c_3" => use_fit_parameters ? bounded_edge_σ : bounded_edge_σ,

    "geometric_ice_c_1" => use_fit_parameters ? bounded_edge_σ : bounded_edge_σ,
    "geometric_ice_c_2" => use_fit_parameters ? bounded_σ : bounded_σ,
    "geometric_ice_c_3" => use_fit_parameters ? bounded_edge_σ : bounded_edge_σ,

    "exponential_T_scaling_ice_c_1" => use_fit_parameters ? bounded_edge_σ : bounded_edge_σ,
    "exponential_T_scaling_ice_c_2" => use_fit_parameters ? FT(1.0) : FT(1.0),

    "powerlaw_T_scaling_ice_c_1" => use_fit_parameters ? FT(10) : FT(10),
    "powerlaw_T_scaling_ice_c_2" => use_fit_parameters ? FT(1) : FT(1),

    "exponential_T_scaling_and_geometric_ice_c_1" => use_fit_parameters ? bounded_edge_σ : bounded_edge_σ,
    "exponential_T_scaling_and_geometric_ice_c_2" => use_fit_parameters ? bounded_σ : bounded_σ,
    "exponential_T_scaling_and_geometric_ice_c_3" => use_fit_parameters ? bounded_edge_σ : bounded_edge_σ,
    "exponential_T_scaling_and_geometric_ice_c_4" => use_fit_parameters ? bounded_edge_σ : bounded_edge_σ,

    ## NEED TO EDIT THE PRIORS ON THESE TO MATCH THE PRIORS/CONSTRAINTS WE'VE CHOSEN -- NOTE NO_CONSTRAINT() WOULD HAVE NO LOG SCALING, IN WHICH CASE WE COULD TAKE THE LOG OF THE PARAMETERS PERHAPS IF NECESSARY
    "linear_combination_liq_c_1" => use_fit_parameters ? unbounded_σ : unbounded_σ,
    "linear_combination_liq_c_2" => use_fit_parameters ? FT(1) : FT(1),
    "linear_combination_liq_c_3" => use_fit_parameters ? unbounded_σ : unbounded_σ,
    "linear_combination_liq_c_4" => use_fit_parameters ? unbounded_σ : unbounded_σ,

    "linear_combination_ice_c_1" => use_fit_parameters ? unbounded_σ : unbounded_σ,
    "linear_combination_ice_c_2" => use_fit_parameters ? bounded_edge_σ : bounded_edge_σ,
    "linear_combination_ice_c_3" => use_fit_parameters ? unbounded_σ : unbounded_σ,
    "linear_combination_ice_c_4" => use_fit_parameters ? unbounded_σ : unbounded_σ,
)

@warn("NEED to check that the priors and constraints are consistent with the unconstrained_σ values!!!... esp where we have no_constraint(), we will have no log scaling, maybe we should take the log of the parameters in some of those cases...")
@warn("NEED to check that the priors and constraints are consistent with the unconstrained_σ values!!!... esp where we have no_constraint(), we will have no log scaling, maybe we should take the log of the parameters in some of those cases...")
@warn("NEED to check that the priors and constraints are consistent with the unconstrained_σ values!!!... esp where we have no_constraint(), we will have no log scaling, maybe we should take the log of the parameters in some of those cases...")
@warn("NEED to check that the priors and constraints are consistent with the unconstrained_σ values!!!... esp where we have no_constraint(), we will have no log scaling, maybe we should take the log of the parameters in some of those cases...")

# --------------------------------------------------------------------------------------------- #

_max_area = FT(0.3)
global_param_defaults["surface_area"] = Dict("prior_mean" =>  FT(default_params["surface_area"]["value"]), "constraints" => bounded(FT(0.0), _max_area),  "unconstrained_σ" => non_vec_sigma) # 0.3 is the max area we want to allow

# --------------------------------------------------------------------------------------------- #
global_param_defaults["pow_icenuc"] = Dict("prior_mean" => FT(default_params["pow_icenuc"]["value"]), "constraints" => bounded_below(0), "unconstrained_σ" => FT(1.0)) # pow_icenuc really shouldn't vary that far...

global_param_defaults["τ_cond_evap"] = Dict("prior_mean" => FT(100.), "constraints" => bounded_below(0), "unconstrained_σ" => expanded_unconstrained_σ ) # a little higher so we can spread it out more relative to ice and allow it to survive even w/ precip

# global_param_defaults["τ_sub_dep"] = Dict("prior_mean" => FT(default_params["τ_sub_dep"]["value"]), "constraints" => bounded_below(0), "unconstrained_σ" => expanded_unconstrained_σ ) # a little higher so we can spread it out more relative to ice and allow it to survive even w/ precip
global_param_defaults["τ_sub_dep"] = Dict("prior_mean" => FT(1000.), "constraints" => bounded_below(0), "unconstrained_σ" => expanded_unconstrained_σ ) # a little higher so we can spread it out more relative to ice and allow it to survive even w/ precip

for var ∈ ["geometric_liq_c_1", "geometric_liq_c_2", "geometric_liq_c_3", "geometric_ice_c_1", "geometric_ice_c_2", "geometric_ice_c_3", "exponential_T_scaling_ice_c_1", "exponential_T_scaling_ice_c_2", "powerlaw_T_scaling_ice_c_1", "powerlaw_T_scaling_ice_c_2", "exponential_T_scaling_and_geometric_ice_c_1", "exponential_T_scaling_and_geometric_ice_c_2", "exponential_T_scaling_and_geometric_ice_c_3", "exponential_T_scaling_and_geometric_ice_c_4", "linear_combination_liq_c_1", "linear_combination_liq_c_2", "linear_combination_liq_c_3", "linear_combination_liq_c_4", "linear_combination_ice_c_1", "linear_combination_ice_c_2", "linear_combination_ice_c_3", "linear_combination_ice_c_4"]
    global_param_defaults[var] = Dict("prior_mean" => priors[var] , "constraints" => constraints[var] , "unconstrained_σ" => unconstrained_σ[var])
end

# global_param_defaults["var"] = Dict("prior_mean" => priors["var"] , "constraints" => constraints["var"]   , "unconstrained_σ" => unconstrained_σ["var"])
# global_param_defaults["geometric_liq_c_2"] = Dict("prior_mean" => priors["geometric_liq_c_2"] , "constraints" => bounded(FT(1/3), FT(1))   , "unconstrained_σ" => bounded_σ)
# global_param_defaults["geometric_liq_c_3"] = Dict("prior_mean" => priors["geometric_liq_c_3"] , "constraints" => bounded_below(FT(0))      , "unconstrained_σ" => bounded_edge_σ)

# global_param_defaults["geometric_ice_c_1"] = Dict("prior_mean" => priors["geometric_ice_c_1"] , "constraints" => bounded_below(FT(0))      , "unconstrained_σ" => bounded_edge_σ)
# global_param_defaults["geometric_ice_c_2"] = Dict("prior_mean" => priors["geometric_ice_c_2"] , "constraints" => bounded(FT(1/3), FT(1))   , "unconstrained_σ" => bounded_σ)
# global_param_defaults["geometric_ice_c_3"] = Dict("prior_mean" => priors["geometric_ice_c_3"] , "constraints" => bounded_below(FT(0))      , "unconstrained_σ" => bounded_edge_σ)


# global_param_defaults["exponential_T_scaling_ice_c_1"] = Dict("prior_mean" => priors["exponential_T_scaling_ice_c_1"] , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ)
# global_param_defaults["exponential_T_scaling_ice_c_2"] = Dict("prior_mean" => priors["exponential_T_scaling_ice_c_2"] , "constraints" => bounded_above(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => FT(1.0)) # exponent has a pretty small range

# global_param_defaults["powerlaw_T_scaling_ice_c_1"] = Dict("prior_mean" => priors["powerlaw_T_scaling_ice_c_1"] , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => FT(10)) # normalizing factor has a pretty large range, 10^0 to 10^20 is realistic perhaps, a lot can be made up with the exponent c_2
# global_param_defaults["powerlaw_T_scaling_ice_c_2"] = Dict("prior_mean" => priors["powerlaw_T_scaling_ice_c_2"] , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => FT(1)) # exponent has small range 

# global_param_defaults["exponential_T_scaling_and_geometric_ice_c_1"] = Dict("prior_mean" => priors["exponential_T_scaling_and_geometric_ice_c_1"] , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ) # Yeahhh.... idk for this one lol... just combined them serially from the homogenous case where c_3 is -1/3
# global_param_defaults["exponential_T_scaling_and_geometric_ice_c_2"] = Dict("prior_mean" => priors["exponential_T_scaling_and_geometric_ice_c_2"] , "constraints" => bounded(1/3, 1)  , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_σ)  # Halfway between 1/3 and 1 -- should this be the same as c_2g? It's the same mixing...
# global_param_defaults["exponential_T_scaling_and_geometric_ice_c_3"] = Dict("prior_mean" => priors["exponential_T_scaling_and_geometric_ice_c_3"] , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ) #  Fletcher 1962 (values taken from Frostenberg 2022)
# global_param_defaults["exponential_T_scaling_and_geometric_ice_c_4"] = Dict("prior_mean" => priors["exponential_T_scaling_and_geometric_ice_c_4"] , "constraints" => bounded_above(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ) # Fletcher 1962 (values taken from Frostenberg 2022), same sign again I suppose...

# global_param_defaults["linear_combination_liq_c_1"] = Dict("prior_mean" => priors["linear_combination_liq_c_1"], "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ) # I think at q=0, we need c_1 from linear = c_1 from geometric...
# global_param_defaults["linear_combination_liq_c_2"] = Dict("prior_mean" => priors["linear_combination_liq_c_2"], "constraints" => no_constraint()  , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => expanded_unbounded_σ) # T scaling on liq uncertain, slower bc favor ice? idk... slower bc loses to PSACWI?
# global_param_defaults["linear_combination_liq_c_3"] = Dict("prior_mean" => priors["linear_combination_liq_c_3"], "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => expanded_bounded_edge_σ) # should have τ down as q up, so start positive
# global_param_defaults["linear_combination_liq_c_4"] = Dict("prior_mean" => priors["linear_combination_liq_c_4"], "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ) # w up should lead to τ down, so coefficient is (probably) positive  # start away from 0 bc that's ∞ in unbounded space
# #
# global_param_defaults["linear_combination_ice_c_1"] = Dict("prior_mean" => priors["linear_combination_ice_c_1"], "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ) # I think at q=0, we need c_1 from linear = c_1 from geometric...
# global_param_defaults["linear_combination_ice_c_2"] = Dict("prior_mean" => priors["linear_combination_ice_c_2"], "constraints" => bounded_above(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => FT(1.0)) # Fletcher 1962 (values taken from Frostenberg 2022), same sign again I suppose...
# global_param_defaults["linear_combination_ice_c_3"] = Dict("prior_mean" => priors["linear_combination_ice_c_3"], "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => expanded_bounded_edge_σ) # should have τ down as q up, so start positive
# global_param_defaults["linear_combination_ice_c_4"] = Dict("prior_mean" => priors["linear_combination_ice_c_4"], "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_edge_σ) #  w up should lead to τ down, so coefficient is positive
# # --------------------------------------------------------------------------------------------- #


q_r_1 = FT((4/3) * π * r_r^3 * 1000) # mass of one raindrop, q = N q_r
B = FT(100) # Hong and Lim 2006 [ Constant in raindrop freezing equation ]
global_param_defaults["heterogeneous_ice_nuclation_coefficient"] = Dict("prior_mean" => q_r_1/ρ_l * B , "constraints" => nothing, "unconstrained_σ" => NaN) # See the writeup
global_param_defaults["heterogeneous_ice_nuclation_exponent"]    = Dict("prior_mean" => FT(0.66)      , "constraints" => nothing, "unconstrained_σ" => NaN) # Hong & Lim 2006 [Constant in Biggs Freezing]

# global_param_defaults["r_ice_snow"] = Dict("prior_mean" => FT(62.5e-6), "constraints" => bounded(0, 1e-3), "unconstrained_σ" => bounded_σ) # 62.5 microns is the default from ClimaParameters
global_param_defaults["r_ice_snow"] = Dict("prior_mean" => default_params["r_ice_snow"]["value"], "constraints" => bounded(0, 1e-3), "unconstrained_σ" => bounded_σ) # 62.5 microns is the default from ClimaParameters

# ========================================================================================================================= #
# ========================================================================================================================= #

path_to_Costa_SOTA = "/groups/esm/cchristo/cedmf_results/james_v1_runs/results_Inversion_p22_e300_i15_mb_LES_2024-03-15_10-08_Vxo_longer_long_run/Diagnostics.nc"
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
    "q_liq_threshold" => Dict("prior_mean" => FT(1e-5) , "constraints" => bounded(0, 5e-4) , "l2_reg" => nothing, "CLIMAParameters_longname" => "cloud_liquid_water_specific_humidity_autoconversion_threshold", "unconstrained_σ" => unconstrained_σ_autoconversion_threshold), # unrealistically high upper bound
    "q_ice_threshold" => Dict("prior_mean" => FT(1e-8) , "constraints" => bounded(0, 1e-6) , "l2_reg" => nothing, "CLIMAParameters_longname" => "cloud_ice_specific_humidity_autoconversion_threshold", "unconstrained_σ" => unconstrained_σ_autoconversion_threshold),          # unrealistically high upper bound
)

sedimentation_parameters = Dict(
    # liq_sedimentation_scaling_factor = Dict("prior_mean" => FT(1.0) , "constraints" => bounded(0,5) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_σ), # not testing rn
    "rain_sedimentation_scaling_factor" => Dict("prior_mean" => FT(1.0) , "constraints" => bounded(0, 5) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_σ), # not testing rn
    "ice_sedimentation_scaling_factor"  => Dict("prior_mean" => FT(1.0) , "constraints" => bounded(0, 5) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_σ), # testing
    "snow_sedimentation_scaling_factor" => Dict("prior_mean" => FT(1.0) , "constraints" => bounded(0, 5) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => bounded_σ), # testing
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
    "surface_area" => Dict("prior_mean" => global_param_defaults["surface_area"]["prior_mean"], "constraints" => global_param_defaults["surface_area"]["constraints"], "l2_reg" => nothing, "CLIMAParameters_longname" => "surface_area", "unconstrained_σ" => global_param_defaults["surface_area"]["unconstrained_σ"]),
    #
    "r_ice_snow" => Dict("prior_mean" => global_param_defaults["r_ice_snow"]["prior_mean"]    , "constraints" => global_param_defaults["r_ice_snow"]["constraints"] , "l2_reg" => nothing, "CLIMAParameters_longname" => "r_ice_snow", "unconstrained_σ" => global_param_defaults["r_ice_snow"]["unconstrained_σ"]),
    # "adjusted_ice_N_scaling_factor" => Dict("prior_mean" => FT(1.0) , "constraints" => bounded_below(0) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => FT(1)), # .1 to 10 should be reasonable lol

)



default_calibration_parameters = merge(default_calibration_parameters, autoconversion_calibration_parameters, sedimentation_parameters)


# Thing's were setting and not calibrating
default_namelist_args = [ # from Costa, i think just starting here is fine and if we overwrite things later they'll just overwrite in the namelist too?
    # ("time_stepping", "dt_min", 0.5),
    # ("time_stepping", "dt_max", 5.0),
    # ("stats_io", "frequency", 60.0),
    # ("time_stepping", "t_max", 3600.0 * SCM_RUN_TIME_HR),
    # ("t_interval_from_end_s", 3600.0 * SCM_RUN_TIME_HR),
    ("thermodynamics", "sgs", "mean"), # quadrature doesn't work w/ noneq
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
    ("grid", "z_reduction_factor", 1), # probably better to change this to have some min dz since this reduction could be really overkill on a stretched/squeezed grid
]


# TEST
default_namelist_args = [ # from Costa, i think just starting here is fine and if we overwrite things later they'll just overwrite in the namelist too?
    # ("time_stepping", "dt_min", 0.5),
    # ("time_stepping", "dt_max", 5.0),
    # ("stats_io", "frequency", 60.0),
    # ("time_stepping", "t_max", 3600.0 * SCM_RUN_TIME_HR),
    # ("t_interval_from_end_s", 3600.0 * SCM_RUN_TIME_HR),
    ("thermodynamics", "sgs", "mean"), # quadrature doesnt work w/ noneq
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
    ("microphysics", "r_ice_snow", default_calibration_parameters["r_ice_snow"]["prior_mean"] ),
    ("user_aux", "rain_sedimentation_scaling_factor", default_calibration_parameters["rain_sedimentation_scaling_factor"]["prior_mean"] ),
    ("user_aux", "ice_sedimentation_scaling_factor", default_calibration_parameters["ice_sedimentation_scaling_factor"]["prior_mean"] ),
    ("user_aux", "snow_sedimentation_scaling_factor", default_calibration_parameters["snow_sedimentation_scaling_factor"]["prior_mean"] ),
    # ("user_aux", "adjusted_ice_N_scaling_factor", default_calibration_parameters["adjusted_ice_N_scaling_factor"]["prior_mean"] ), # this should just be constants in the thing right
    ("user_aux", "ice_sedimentation_Dmax", FT(Inf)), # 62.5 microns cutoff from CM
    # ("grid", "z_reduction_factor", 1),
    ("grid", "dz_min", FT(10)),
    ("user_aux", "min_τ_liq", FT(1e-2)), # For stability in timestepping and removing buoayancy shockwaves etc
    ("user_aux", "min_τ_ice", FT(1e-2)), # For stability in timestepping and remove buoyancy shockwaves etc
    ("time_stepping", "spinup_dt_factor", FT(0.25)), # not sure if these should go here or in get_scm_config() from footer()
    ("time_stepping", "spinup_half_t_max", FT(3600.0 * 1.0)), # 1 hour
    ("time_stepping", "spinup_adapt_dt", false),
]

default_namelist_args = [Costa_SOTA_namelist_args; default_namelist_args] # add the Costa SOTA namelist args to the default ones

# ========================================================================================================================= #
# can we add a default user_args that we add to/merge with later in _footer.jl? yes, it's in _footer.jl
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
    "qip_mean"         => obs_var_additional_uncertainty_factor,
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
    "qip_mean"         => FT(1e-3), # 1e-3 kg/kg
    "ql_all_mean"      => FT(1e-3), # 1e-3 kg/kg
    "qi_all_mean"      => FT(1e-8), # 1e-3 kg/kg
    "qt_mean"          => FT(1e-3), # 1e-3 kg/kg
)


default_obs_var_scaling = Dict{String, FT}(
    "temperature_mean" => (1/273.0)^2, #  scale down bc we scaled T to 1, so ΔT is now ∼ 1/273, so scale that up, leave others the same. then ΔT will be O(1) just like Δq
    "ql_mean" => FT((1.0/2)^2), # boost by factor of 2
    "qi_mean" => FT((1.0/5)^2), # boost by factor of 5
    "qr_mean" => FT(1.0),
    "qs_mean" => FT(1.0),
    "qip_mean" => FT(1.0),
    "ql_all_mean" => FT(1.0), # leave equal
    "qi_all_mean" => FT(1.0), # leave equal
    
    ) # scale down so ice becomes more important (factor of 5 rn), maybe will help calibrations...
    
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
# additive_inflation = nothing # default to not using it
additive_inflation = 1e-8 # default to not using it

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
        13 => (missing, missing),
        )
)

