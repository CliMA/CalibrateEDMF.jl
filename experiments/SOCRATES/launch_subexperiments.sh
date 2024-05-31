#!/bin/bash

# Don't forget to change to 14 hours one day

# SOCRATES_Base
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_Base tau_autoconv_noneq
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_Base pow_icenuc_noneq

# SOCRATES_geometric_liq__geometric_ice
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_geometric_liq__geometric_ice tau_autoconv_noneq

# SOCRATES_geometric_liq__exponential_T_scaling
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_geometric_liq__exponential_T_scaling tau_autoconv_noneq

# SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice tau_autoconv_noneq

# SOCRATES_linear_combination
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_linear_combination tau_autoconv_noneq

# SOCRATES_linear_combination_with_w
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_linear_combination_with_w tau_autoconv_noneq

# SOCRATES_neural_network
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/global_calibrate.sh SOCRATES_neural_network tau_autoconv_noneq

# restart (edit to use)
clear; sbatch -p expansion ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/restart_global_calibrate.sh SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice tau_autoconv_noneq




send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_Base
send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_geometric_liq__geometric_ice
send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_geometric_liq__exponential_T_scaling
send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice
send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_linear_combination
send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_linear_combination_with_w
send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/SOCRATES_neural_network

# send_to_sampo ~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES


