#!/bin/bash
#Submit this script with: sbatch calibrate_script
#SBATCH --time=00:01:00   # walltime
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "launch_calibrate"   # job name
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH --output=/dev/null

experiment=${1?Error: no experiment given}
calibration_setup=${2?Error: no calibration setup given} # the subname of the calibration setup within the experiment (these should almost just be different experiments tbh, but you could still i guess toggle doing autoconv or no, or calibrate to LES or flight obs, or use ERA5 or flight_obs LES runs.
calibration_vars_str=${3?Error: no calibration vars given}

if [ -z "$4" ]; then
    # echo "max_preconditioner_counter not set, defaulting to 10"
    max_preconditioner_counter=10
else
    max_preconditioner_counter=$4
fi

calibrate_script=~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/global_parallel/ekp_par_calibration.sbatch
experiment_path=~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/subexperiments/${experiment}/
calibrate_and_run_dir=$experiment_path/Calibrate_and_Run/
this_calibrate_and_run=$calibration_setup/$calibration_vars_str
this_calibrate_and_run_dir=$calibrate_and_run_dir/$this_calibrate_and_run/
this_calibrate_dir=$this_calibrate_and_run_dir/calibrate/
this_config_dir=$this_calibrate_dir/configs/
log_dir=$this_calibrate_dir/logs/

echo $calibrate_script
echo $this_config_dir

# Calibrate tau + autoconv params | eq

use_expansion=true


if [ "$use_expansion" = false ]; then

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
        true; sbatch -o ${log_dir}/RFAll_obs.out $calibrate_script ${this_config_dir}/config_calibrate_RFAll_obs.jl $use_expansion $max_preconditioner_counter
    else
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF01_obs.jl  > ${log_dir}/RF01_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF09_obs.jl  > ${log_dir}/RF09_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF10_obs.jl  > ${log_dir}/RF10_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF11_obs.jl  > ${log_dir}/RF11_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF12_obs.jl  > ${log_dir}/RF12_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF13_obs.jl  > ${log_dir}/RF13_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_RFAll_obs.jl > ${log_dir}/RFAll_obs.out
        :
    fi
    # Possible Issues : calibrate_script timed out last time.

else

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
        true; sbatch -o ${log_dir}/RFAll_obs.out --partition=expansion $calibrate_script ${this_config_dir}/config_calibrate_RFAll_obs.jl $use_expansion $max_preconditioner_counter
    else
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF01_obs.jl  > ${log_dir}/RF01_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF09_obs.jl  > ${log_dir}/RF09_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF10_obs.jl  > ${log_dir}/RF10_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF11_obs.jl  > ${log_dir}/RF11_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF12_obs.jl  > ${log_dir}/RF12_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_${this_calibrate_and_run}_RF13_obs.jl  > ${log_dir}/RF13_obs.out
        # clear; sh $calibrate_script  ${this_config_dir}/config_calibrate_RFAll_obs.jl > ${log_dir}/RFAll_obs.out
        :
    fi
    # Possible Issues : calibrate_script timed out last time.

fi