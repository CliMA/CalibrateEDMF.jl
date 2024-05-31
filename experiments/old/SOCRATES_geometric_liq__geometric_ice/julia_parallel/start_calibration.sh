#!/bin/bash

thisfile="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")" # the location of this file I think is more versatile than the stack overflow solution (see https://stackoverflow.com/a/9107028/13331585), realpath might be better though idk...

# check if script is started via SLURM or bash (https://stackoverflow.com/q/56962129/13331585) | if with SLURM: there variable '$SLURM_JOB_ID' will exist `if [ -n $SLURM_JOB_ID ]` checks if $SLURM_JOB_ID is not an empty string
if [ -n "$SLURM_JOB_ID" ];  then # need to add quotes here, see https://unix.stackexchange.com/questions/109625/shell-scripting-z-and-n-options-with-if#comment1152278_109631
    # check the original location through scontrol and $SLURM_JOB_ID
    thisfile=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else # otherwise: started with bash. Get the real location.
    # thisfile=$(realpath $0)
    thisfile="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")" # the location of this file I think is more versatile than the stack overflow solution (see https://stackoverflow.com/a/9107028/13331585), realpath might be better though idk...
fi
thisdir=$(dirname $thisfile) # go up to directory from filename...
experiment_dir=$(dirname $thisdir) # go up to experiment directory...
log_dir=${experiment_dir}/Output/Logs/
config=${1?Error: no config file given} # maybe we add later to just default to config.jl in the experiment directory?
# config_file=$experiment_dir/config.jl

# Pass on the calling method for this script...
if [ -n "$SLURM_JOB_ID" ];  then # need to add quotes here, see https://unix.stackexchange.com/questions/109625/shell-scripting-z-and-n-options-with-if#comment1152278_109631
    sbatch --output ${experiment_dir}/Output/Logs/slurm_julia_par_%j.out $thisdir/calibrate_script $config # Not sure how many processors I should go for... maybe at least 1 more than the batch size?
else
    sh $thisdir/calibrate_script $config # Not sure how many processors I should go for... maybe at least 1 more than the batch size?
fi


