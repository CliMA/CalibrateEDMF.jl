# Perform a single step of the EnsembleKalmanProcess

# Import modules to all processes
using ArgParse
using Distributions
using StatsBase
using LinearAlgebra
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
include(joinpath(@__DIR__, "../../../src/helper_funcs.jl"))
using JLD2
using Base


function ek_update(iteration_::Int64, outdir_path::String)
    # Recover versions from last iteration
    versions = readlines(joinpath(outdir_path, "versions_$(iteration_).txt"))
    ekobj = deserialize_struct(load(ekobj_path(outdir_path, iteration_))["ekp"], EnsembleKalmanProcess)
    N_par, N_ens = size(ekobj.u[1])
    @assert N_ens == length(versions)

    scm_args = load(scm_init_path(outdir_path, versions[1]))
    ref_stats = deserialize_struct(scm_args["ref_stats"], ReferenceStatistics)
    ref_models = map(x -> deserialize_struct(x, ReferenceModel), scm_args["ref_models"])
    u = zeros(N_ens, N_par)
    g = zeros(N_ens, pca_length(ref_stats))
    for (ens_index, version) in enumerate(versions)
        scm_args = load(scm_init_path(outdir_path, version))
        scm_outputs = load(scm_output_path(outdir_path, version))
        u[ens_index, :] = scm_args["u"]
        g[ens_index, :] = scm_outputs["g_scm_pca"]
    end
    g = Array(g')
    # Advance EKP
    update_ensemble!(ekobj, g)
    ekp = serialize_struct(ekobj)
    jldsave(ekobj_path(outdir_path, iteration_ + 1); ekp)

    # Get new step
    priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))
    params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekobj))
    params = [c[:] for c in eachcol(params_cons_i)]
    versions = map(param -> generate_scm_input(param, priors.names, ref_models, ref_stats, outdir_path), params)
    # Store version identifiers for this ensemble in a common file
    open(joinpath(outdir_path, "versions_$(iteration_+1).txt"), "w") do io
        for version in versions
            write(io, "$(version)\n")
        end
    end
    return
end


# Read iteration number of ensemble to be recovered
s = ArgParseSettings()
@add_arg_table s begin
    "--iteration"
    help = "Calibration iteration number"
    arg_type = Int
end
@add_arg_table s begin
    "--job_dir"
    help = "Job output directory"
    arg_type = String
    default = "output"
end
parsed_args = parse_args(ARGS, s)
ek_update(parsed_args["iteration"], parsed_args["job_dir"])
