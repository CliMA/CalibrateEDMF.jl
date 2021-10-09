# Evaluates a set of SCM configurations for a single parameter vector.
#
using ArgParse
using Distributions
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
include(joinpath(@__DIR__, "../../../src/helper_funcs.jl"))
using JLD2


# Read iteration number of ensemble to be recovered
s = ArgParseSettings()
@add_arg_table s begin
    "--version"
    help = "Calibration process number"
    arg_type = Int
end
@add_arg_table s begin
    "--job_dir"
    help = "Job output directory"
    arg_type = String
    default = "output"
end
parsed_args = parse_args(ARGS, s)
version = parsed_args["version"]
outdir_path = parsed_args["job_dir"]

scm_args = load(scm_init_path(outdir_path, version))

sim_dirs, g_scm, g_scm_pca = run_SCM(
    scm_args["u"],
    scm_args["u_names"],
    map(x -> deserialize_struct(x, ReferenceModel), scm_args["ref_models"]),
    deserialize_struct(scm_args["ref_stats"], ReferenceStatistics),
)

jldsave(scm_output_path(outdir_path, version); sim_dirs, g_scm, g_scm_pca, version)
