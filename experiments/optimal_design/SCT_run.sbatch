#!/bin/bash

#SBATCH --time=2:00:00   # walltime
#SBATCH --ntasks=10  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "hadgem"   # job name

module load julia/1.9.3 openmpi/4.1.5

julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

#### PERFECT MODEL DEFINITION
# Define the root directory of the calibration experiments from which to retrieve true parameters
exp_dir=/groups/esm/ilopezgo/CalibrateEDMF.jl/experiments/optimal_design/global_parallel/
# Choose retrieval method -- in this case, take the nearest neighbor to the mean at last iteration
method=last_nn_particle_mean
# Define the calibration directory from which to draw the parameters
experiment=results_Inversion_dt_1.0_p2_e50_i10_d40_LES_2022-07-13_22-31_ewR

#### OUTPUT AND DATA DEFINITION
# Define the output directory where output data will be stored
out_name=perf_model_HadGEM2_nz55_B38_Inv_d40_p2_ewR
# Path to TCRunner
tc_runner_path=../../tools/
# Path to config defining the data to be generated, RELATIVE TO TCRunner!
test_data_config=../experiments/optimal_design/hadgem_amip_sct_config.jl


# Do not modify
tc_runner=TCRunner.jl

#######   RUN SCRIPTS   #######
# # Generate HadGEM2A AMIP data
### This assumes that 
julia -p10 $tc_runner_path$tc_runner --results_dir=$exp_dir$experiment --tc_output_dir=$out_name --method=$method --run_set="test" --run_set_config=$test_data_config
cp $tc_runner_path$test_data_config $out_name/data_generator_config.jl

cp $exp_dir$experiment/config.jl $out_name/calibrated_model_config.jl
