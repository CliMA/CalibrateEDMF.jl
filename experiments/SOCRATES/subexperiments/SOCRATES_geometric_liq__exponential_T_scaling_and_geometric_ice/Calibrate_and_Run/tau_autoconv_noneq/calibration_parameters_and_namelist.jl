#=
This file is meant to be included into a larger framework so intermediate variables may be undefined
=#

# ========================================================================================================================= #
# constants we use here
ρ_l = FT(1000.0) # density of ice, default from ClimaParameters
ρ_i = FT(916.7) # density of ice, default from ClimaParameters
r_r = FT(20 * 1e-6) # 20 microns
r_0 = FT(0.2 * 1e-6) # .2 micron base aerosol

N_l = FT(1e-5 / (4 / 3 * π * r_r^3 * ρ_l)) # estimated total N assuming reasonable q_liq.. (N = N_r in homogenous)
N_i = FT(1e-7 / (4 / 3 * π * r_r^3 * ρ_i)) # estimated total N assuming reasonable q_ice... (N = N_r + N_0)

D_ref = FT(0.0000226)
D = D_ref

# ========================================================================================================================= #

calibration_parameters__experiment_setup = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    #
    "geometric_liq_c_1" => Dict(
        "prior_mean" => global_param_defaults["geometric_liq_c_1"]["prior_mean"],
        "constraints" => global_param_defaults["geometric_liq_c_1"]["constraints"],
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => nothing,
        "unconstrained_σ" => global_param_defaults["geometric_liq_c_1"]["unconstrained_σ"],
    ),
    "geometric_liq_c_2" => Dict(
        "prior_mean" => global_param_defaults["geometric_liq_c_2"]["prior_mean"],
        "constraints" => global_param_defaults["geometric_liq_c_2"]["constraints"],
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => nothing,
        "unconstrained_σ" => global_param_defaults["geometric_liq_c_2"]["unconstrained_σ"],
    ),
    "geometric_liq_c_3" => Dict(
        "prior_mean" => global_param_defaults["geometric_liq_c_3"]["prior_mean"],
        "constraints" => global_param_defaults["geometric_liq_c_3"]["constraints"],
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => nothing,
        "unconstrained_σ" => global_param_defaults["geometric_liq_c_3"]["unconstrained_σ"],
    ),
    #
    "exponential_T_scaling_and_geometric_ice_c_1" => Dict(
        "prior_mean" => global_param_defaults["exponential_T_scaling_and_geometric_ice_c_1"]["prior_mean"],
        "constraints" => global_param_defaults["exponential_T_scaling_and_geometric_ice_c_1"]["constraints"],
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => nothing,
        "unconstrained_σ" =>
            global_param_defaults["exponential_T_scaling_and_geometric_ice_c_1"]["unconstrained_σ"],
    ),
    "exponential_T_scaling_and_geometric_ice_c_2" => Dict(
        "prior_mean" => global_param_defaults["exponential_T_scaling_and_geometric_ice_c_2"]["prior_mean"],
        "constraints" => global_param_defaults["exponential_T_scaling_and_geometric_ice_c_2"]["constraints"],
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => nothing,
        "unconstrained_σ" =>
            global_param_defaults["exponential_T_scaling_and_geometric_ice_c_2"]["unconstrained_σ"],
    ),
    "exponential_T_scaling_and_geometric_ice_c_3" => Dict(
        "prior_mean" => global_param_defaults["exponential_T_scaling_and_geometric_ice_c_3"]["prior_mean"],
        "constraints" => global_param_defaults["exponential_T_scaling_and_geometric_ice_c_3"]["constraints"],
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => nothing,
        "unconstrained_σ" =>
            global_param_defaults["exponential_T_scaling_and_geometric_ice_c_3"]["unconstrained_σ"],
    ),
    "exponential_T_scaling_and_geometric_ice_c_4" => Dict(
        "prior_mean" => global_param_defaults["exponential_T_scaling_and_geometric_ice_c_4"]["prior_mean"],
        "constraints" => global_param_defaults["exponential_T_scaling_and_geometric_ice_c_4"]["constraints"],
        "l2_reg" => nothing,
        "CLIMAParameters_longname" => nothing,
        "unconstrained_σ" =>
            global_param_defaults["exponential_T_scaling_and_geometric_ice_c_4"]["unconstrained_σ"],
    ),
)

local_namelist__experiment_setup = [ # things in namelist that otherwise wouldn't be... (both random parameters and parameters we want to calibrate that we added ourselves that generate_namelist doesn't insert...)
    ("user_aux", "geometric_liq_c_1", calibration_parameters__experiment_setup["geometric_liq_c_1"]["prior_mean"]),
    ("user_aux", "geometric_liq_c_2", calibration_parameters__experiment_setup["geometric_liq_c_2"]["prior_mean"]),
    ("user_aux", "geometric_liq_c_3", calibration_parameters__experiment_setup["geometric_liq_c_3"]["prior_mean"]),
    #
    (
        "user_aux",
        "exponential_T_scaling_and_geometric_ice_c_1",
        calibration_parameters__experiment_setup["exponential_T_scaling_and_geometric_ice_c_1"]["prior_mean"],
    ),
    (
        "user_aux",
        "exponential_T_scaling_and_geometric_ice_c_2",
        calibration_parameters__experiment_setup["exponential_T_scaling_and_geometric_ice_c_2"]["prior_mean"],
    ),
    (
        "user_aux",
        "exponential_T_scaling_and_geometric_ice_c_3",
        calibration_parameters__experiment_setup["exponential_T_scaling_and_geometric_ice_c_3"]["prior_mean"],
    ),
    (
        "user_aux",
        "exponential_T_scaling_and_geometric_ice_c_4",
        calibration_parameters__experiment_setup["exponential_T_scaling_and_geometric_ice_c_4"]["prior_mean"],
    ),
    #
]
