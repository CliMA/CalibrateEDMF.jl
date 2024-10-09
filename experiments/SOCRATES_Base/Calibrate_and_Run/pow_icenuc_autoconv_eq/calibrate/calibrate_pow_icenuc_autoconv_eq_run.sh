#!/bin/bash
#Submit this script with: sbatch calibrate_script
#SBATCH --time=00:01:00   # walltime
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "launch_calibrate_pow_icenuc_autoconv_eq"   # job name
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH --output=/dev/null


calibrate_script=~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Test_Dynamical_Calibration/julia_parallel/calibrate_script # update these to work form anywhere so can use HPC too... see start_calibration.sh
experiment_path=~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Test_Dynamical_Calibration
calibrate_and_run_dir=$experiment_path/Calibrate_and_Run/
this_calibrate_and_run=pow_icenuc_autoconv_eq
this_calibrate_and_run_dir=$calibrate_and_run_dir/$this_calibrate_and_run/
this_calibrate_dir=$this_calibrate_and_run_dir/calibrate/
this_config_dir=$this_calibrate_dir/configs/
log_dir=$this_calibrate_dir/logs/

# Calibrate ramp (pow_icenuc) + autoconv params | eq

# Pass on the calling method for this script... Ican i mke this a loop over an array?
if true;  then # need to add quotes here, see https://unix.stackexchange.com/questions/109625/shell-scripting-z-and-n-options-with-if#comment1152278_109631
# if false;  then # need to add quotes here, see https://unix.stackexchange.com/questions/109625/shell-scripting-z-and-n-options-with-if#comment1152278_109631
# if [ -n "$SLURM_JOB_ID" ];  then # need to add quotes here, see https://unix.stackexchange.com/questions/109625/shell-scripting-z-and-n-options-with-if#comment1152278_109631 , doesnt work right if run from an interactive node on hpc, since SLURM_JOB_ID is always set in that case...
    # clear; sbatch -o ${log_dir}/RF01_obs.out  $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF01_obs.jl 
    # clear; sbatch -o ${log_dir}/RF09_obs.out  $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF09_obs.jl
    # clear; sbatch -o ${log_dir}/RF10_obs.out  $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF10_obs.jl
    # clear; sbatch -o ${log_dir}/RF11_obs.out  $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF11_obs.jl
    # clear; sbatch -o ${log_dir}/RF12_obs.out  $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF12_obs.jl
    # clear; sbatch -o ${log_dir}/RF13_obs.out  $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF13_obs.jl
    clear; sbatch -o ${log_dir}/RFAll_obs.out $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RFAll_obs.jl
else
    clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF01_obs.jl  > ${log_dir}/RF01_obs.out
    clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF09_obs.jl  > ${log_dir}/RF09_obs.out
    clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF10_obs.jl  > ${log_dir}/RF10_obs.out
    clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF11_obs.jl  > ${log_dir}/RF11_obs.out
    clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF12_obs.jl  > ${log_dir}/RF12_obs.out
    clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF13_obs.jl  > ${log_dir}/RF13_obs.out
    clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RFAll_obs.jl > ${log_dir}/RFAll_obs.out
fi

# Possible Issues : calibrate_script timed out last time.