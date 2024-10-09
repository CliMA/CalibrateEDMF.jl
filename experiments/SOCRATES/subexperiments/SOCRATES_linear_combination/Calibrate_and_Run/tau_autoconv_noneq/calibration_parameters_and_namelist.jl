#=
This file is meant to be included into a larger framework so intermediate variables may be undefined
=#

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
bounded_σ = FT(2.0) # most of the log space is pressed up against the edges so not too big, assume it's closer to linear than log
bounded_edge_σ = FT(5.0) # is in log space go crazy (10 orders of magnitude is pretty big though)
unbounded_σ = FT(5.0) # is in real space so... good luck lol
expanded_unbounded_σ = FT(1e5) # alternate for unconstrained_σ when we truly don't know the prior mean (or it's very uncertain)  ## TESTING 100 HERE!!!! (seem to have some nan errors...) (bounded_below values are log spaced  so consider that)

calibration_parameters__experiment_setup = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    #
    "linear_combination_liq_c_1" => Dict("prior_mean" => global_param_defaults["linear_combination_liq_c_1"]["prior_mean"], "constraints" => global_param_defaults["linear_combination_liq_c_1"]["constraints"] , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => global_param_defaults["linear_combination_liq_c_1"]["unconstrained_σ"]),
    "linear_combination_liq_c_2" => Dict("prior_mean" => global_param_defaults["linear_combination_liq_c_2"]["prior_mean"], "constraints" => global_param_defaults["linear_combination_liq_c_2"]["constraints"] , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => global_param_defaults["linear_combination_liq_c_2"]["unconstrained_σ"]),
    "linear_combination_liq_c_3" => Dict("prior_mean" => global_param_defaults["linear_combination_liq_c_3"]["prior_mean"], "constraints" => global_param_defaults["linear_combination_liq_c_3"]["constraints"] , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => global_param_defaults["linear_combination_liq_c_3"]["unconstrained_σ"]),
    #
    "linear_combination_ice_c_1" => Dict("prior_mean" => global_param_defaults["linear_combination_ice_c_1"]["prior_mean"], "constraints" => global_param_defaults["linear_combination_ice_c_1"]["constraints"] , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => global_param_defaults["linear_combination_ice_c_1"]["unconstrained_σ"]),
    "linear_combination_ice_c_2" => Dict("prior_mean" => global_param_defaults["linear_combination_ice_c_2"]["prior_mean"], "constraints" => global_param_defaults["linear_combination_ice_c_2"]["constraints"] , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => global_param_defaults["linear_combination_ice_c_2"]["unconstrained_σ"]),
    "linear_combination_ice_c_3" => Dict("prior_mean" => global_param_defaults["linear_combination_ice_c_3"]["prior_mean"], "constraints" => global_param_defaults["linear_combination_ice_c_3"]["constraints"] , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => global_param_defaults["linear_combination_ice_c_3"]["unconstrained_σ"]),
    #
) 

local_namelist__experiment_setup = [ # things in namelist that otherwise wouldn't be... (both random parameters and parameters we want to calibrate that we added ourselves that generate_namelist doesn't insert...)
    ("user_aux", "linear_combination_liq_c_1", calibration_parameters__experiment_setup["linear_combination_liq_c_1"]["prior_mean"] ),
    ("user_aux", "linear_combination_liq_c_2", calibration_parameters__experiment_setup["linear_combination_liq_c_2"]["prior_mean"] ),
    ("user_aux", "linear_combination_liq_c_3", calibration_parameters__experiment_setup["linear_combination_liq_c_3"]["prior_mean"] ),
    # ("user_aux", "linear_combination_liq_c_4", calibration_parameters__experiment_setup["linear_combination_liq_c_4"]["prior_mean"] ), # do we need an offset?
    #
    ("user_aux", "linear_combination_ice_c_1", calibration_parameters__experiment_setup["linear_combination_ice_c_1"]["prior_mean"] ),
    ("user_aux", "linear_combination_ice_c_2", calibration_parameters__experiment_setup["linear_combination_ice_c_2"]["prior_mean"] ),
    ("user_aux", "linear_combination_ice_c_3", calibration_parameters__experiment_setup["linear_combination_ice_c_3"]["prior_mean"] ),
    # ("user_aux", "linear_combination_ice_c_4", calibration_parameters__experiment_setup["linear_combination_ice_c_4"]["prior_mean"] ), # do we need and offset?
    #
]
