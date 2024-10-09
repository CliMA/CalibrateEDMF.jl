#!/bin/bash

# Don't forget to change to 14 hours one day


# new_calibration_vars_list=(["ql_mean", "qi_mean"],["ql_all_mean", "qi_all_mean"],["temperature_mean", "ql_mean", "qi_mean"],["temperature_mean", "ql_all_mean", "qi_all_mean"],) # arrays of arrays don't work in bash


experiments=(
    "SOCRATES_Base",
    "SOCRATES_exponential_T_scaling_ice",
    "SOCRATES_exponential_T_scaling_ice_raw",
    "SOCRATES_powerlaw_T_scaling_ice",
    "SOCRATES_geometric_liq__geometric_ice",
    "SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice",
    "SOCRATES_geometric_liq__powerlaw_T_scaling_ice",
    "SOCRATES_neural_network",
    "SOCRATES_linear_combination",
    "SOCRATES_linear_combination_with_w",
)


setups=("pow_icenuc_autoconv_eq", "tau_autoconv_noneq",)

dt_string = "adapt_dt__dt_min_0.5__dt_max_2.0"
# ========================================================================================================================= #

# SOCRATES_Base
# clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_Base $dt_string pow_icenuc_autoconv_eq 5
# clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_Base $dt_string tau_autoconv_noneq 5

# SOCRATES_geometric_liq__geometric_ice
# clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_geometric_liq__geometric_ice $dt_string tau_autoconv_noneq 5

# SOCRATES_exponential_T_scaling_ice
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_exponential_T_scaling_ice $dt_string tau_autoconv_noneq 5 1 1 1
# clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_exponential_T_scaling_ice_raw $dt_string tau_autoconv_noneq 5
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_powerlaw_T_scaling_ice $dt_string tau_autoconv_noneq 5
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_geometric_liq__powerlaw_T_scaling_ice $dt_string tau_autoconv_noneq 5


clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_geometric_liq__exponential_T_scaling_ice $dt_string tau_autoconv_noneq 5

# SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice $dt_string tau_autoconv_noneq 5

# SOCRATES_linear_combination
# clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_linear_combination $dt_string tau_autoconv_noneq 5

# SOCRATES_linear_combination_with_w
# clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_linear_combination_with_w $dt_string tau_autoconv_noneq qi_mean__qip_mean__ql_mean__qr_mean $dt_string tau_autoconv_noneq 5 1 1 1

# SOCRATES_neural_network
# clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_neural_network $dt_string tau_autoconv_noneq qi_mean__qip_mean__ql_mean__qr_mean $dt_string tau_autoconv_noneq 5 1 1 1


# ========================================================================================================================= #

# restart (edit to use)
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/restart_global_calibrate.sh SOCRATES_neural_network tau_autoconv_noneq  $dt_string qi_mean__qip_mean__ql_mean__qr_mean 5 1 1 1

# ========================================================================================================================= #


send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_Base
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_exponential_T_scaling_ice

# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_geometric_liq__geometric_ice
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_geometric_liq__exponential_T_scaling
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_linear_combination
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_linear_combination_with_w
# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_neural_network

# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES


