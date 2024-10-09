#!/bin/bash
calibrate_script=~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Test_Dynamical_Calibration/julia_parallel/calibrate_script
experiment_path=~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Test_Dynamical_Calibration/
calibrate_and_run_dir=$experiment_path/Calibrate_and_Run/
this_calibrate_and_run=tau_autoconv_noneq
this_calibrate_and_run_dir=$calibrate_and_run_dir/$this_calibrate_and_run/
config_path=$this_calibrate_and_run_dir/calibrate/
log_dir=$this_calibrate_and_run_dir/logs/

# Run the model w/ the calibrated τ values, noneq
clear; sbatch -o ${log_dir}/run_calibrated_RF01_obs.out <run somhow assign tasks sbathc stuff> ${this_config_dir}/run_calibrated_${this_config}_RF01_obs.jl
clear; sbatch -o ${log_dir}/run_calibrated_RF09_obs.out <run somhow assign tasks sbathc stuff> ${this_config_dir}/run_calibrated_${this_config}_RF09_obs.jl
clear; sbatch -o ${log_dir}/run_calibrated_RF10_obs.out <run somhow assign tasks sbathc stuff> ${this_config_dir}/run_calibrated_${this_config}_RF10_obs.jl
clear; sbatch -o ${log_dir}/run_calibrated_RF11_obs.out <run somhow assign tasks sbathc stuff> ${this_config_dir}/run_calibrated_${this_config}_RF11_obs.jl
clear; sbatch -o ${log_dir}/run_calibrated_RF12_obs.out <run somhow assign tasks sbathc stuff> ${this_config_dir}/run_calibrated_${this_config}_RF12_obs.jl
clear; sbatch -o ${log_dir}/run_calibrated_RF13_obs.out <run somhow assign tasks sbathc stuff> ${this_config_dir}/run_calibrated_${this_config}_RF13_obs.jl
clear; sbatch -o ${log_dir}/run_calibrated_RFAll_obs.out <run somhow assign tasks sbathc stuff> ${this_config_dir}/run_calibrated_${this_config}_RFAll_obs.jl

# To Do
- edit so calibration outputs go to somewhere here, not a regular directory in Calibrations...
- edit so running outputs go to somewhere here, not a regular directory in Calibrated_Runs...


