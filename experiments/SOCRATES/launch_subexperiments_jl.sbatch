#!/bin/bash

#SBATCH --time=48:00:00  # walltime (what times do we need for our runs) -- just inits so 20 should be fine
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1        # number of nodes
#SBATCH --mem-per-cpu=1G # is a pretty light job, just launches other sbatch jobs
#SBATCH -J "launch_subexperiments" # job name
#SBATCH -o /home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/launch_subexperiments_jl.out

# This is a controller script for launching jobs and staying below the QOS limit on Caltech HPC...

cd /home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES
julia --project -e 'include("launch_subexperiments.jl")'

# scancel -u jbenjami -n precond; scancel -u jbenjami -n sct_call; scancel -u jbenjami -n sct_run; scancel -u jbenjami -n sct_cont;  scancel -u jbenjami -n sct_init; scancel -u jbenjami -n sct_call; scancel -u jbenjami -n launch_subexperiments

