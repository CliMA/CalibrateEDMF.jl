#!/bin/bash
# You should just submit this script normally, sh start_calibration_script.sh


# thisfile=$(realpath $0)
thisfile="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")" # the location of this file I think is more versatile than the stack overflow solution (see https://stackoverflow.com/a/9107028/13331585), realpath might be better though idk...
thisdir=$(dirname $thisfile) # go up to directory from filename...
experiment_dir=$(dirname $thisdir) # go up to experiment directory...
# Retrieve the configuration file we want to use
config=${1?Error: no config file given} # maybe we add later to just default to config.jl in the experiment directory?
# config_file=$experiment_dir/config.jl

# start the calibration
sbatch $thisdir/ekp_par_calibration.sbatch $config_file # 
