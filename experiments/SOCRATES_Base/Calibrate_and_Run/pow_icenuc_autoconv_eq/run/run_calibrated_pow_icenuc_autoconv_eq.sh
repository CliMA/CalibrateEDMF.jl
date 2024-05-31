run_calibrated_script=~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Test_Dynamical_Calibration/julia_parallel/run_calibrated_script # update these to work form anywhere so can use HPC too... see start_calibration.sh
experiment_path=~/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES_Test_Dynamical_Calibration
calibrate_and_run_dir=$experiment_path/Calibrate_and_Run/
this_calibrate_and_run=pow_icenuc_autoconv_eq
this_calibrate_and_run_dir=$calibrate_and_run_dir/$this_calibrate_and_run/
this_calibrate_dir=$this_calibrate_and_run_dir/run/
this_runscript_dir=$this_calibrate_dir/run_scripts/
log_dir=$this_calibrate_dir/logs/

# Pass on the calling method for this script... Ican i mke this a loop over an array?
if true;  then # need to add quotes here, see https://unix.stackexchange.com/questions/109625/shell-scripting-z-and-n-options-with-if#comment1152278_109631
# if false;  then # need to add quotes here, see https://unix.stackexchange.com/questions/109625/shell-scripting-z-and-n-options-with-if#comment1152278_109631
# if [ -n "$SLURM_JOB_ID" ];  then # need to add quotes here, see https://unix.stackexchange.com/questions/109625/shell-scripting-z-and-n-options-with-if#comment1152278_109631 , doesnt work right if run from an interactive node on hpc, since SLURM_JOB_ID is always set in that case...
# Run the model (eq) with calibrated ramp (pow_icenuc) + autoconv params
    clear; sbatch -o ${log_dir}/RF01_obs.out  $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF01_obs.jl
    clear; sbatch -o ${log_dir}/RF09_obs.out  $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF09_obs.jl
    clear; sbatch -o ${log_dir}/RF10_obs.out  $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF10_obs.jl
    clear; sbatch -o ${log_dir}/RF11_obs.out  $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF11_obs.jl
    clear; sbatch -o ${log_dir}/RF12_obs.out  $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF12_obs.jl
    clear; sbatch -o ${log_dir}/RF13_obs.out  $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF13_obs.jl
    clear; sbatch -o ${log_dir}/RFAll_obs.out $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RFAll_obs.jl
else
    clear; sh $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF01_obs.jl  > ${log_dir}/RF01_obs.out
    clear; sh $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF09_obs.jl  > ${log_dir}/RF09_obs.out
    clear; sh $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF10_obs.jl  > ${log_dir}/RF10_obs.out
    clear; sh $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF11_obs.jl  > ${log_dir}/RF11_obs.out
    clear; sh $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF12_obs.jl  > ${log_dir}/RF12_obs.out
    clear; sh $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RF13_obs.jl  > ${log_dir}/RF13_obs.out
    # clear; sh $run_calibrated_script ${this_runscript_dir}/run_calibrated_${this_calibrate_and_run}_RFAll_obs.jl > ${log_dir}/RFAll_obs.out
fi


