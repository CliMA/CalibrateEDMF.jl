# Initializes a SCM calibration process.

using Distributions
using StatsBase
using LinearAlgebra
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
include(joinpath(@__DIR__, "../../../src/helper_funcs.jl"))
using JLD2

""" Define parameters and their priors"""
function construct_priors(params::Dict{String, Vector{Constraint}}, outdir_path::String = pwd())
    u_names = collect(keys(params))
    constraints = collect(values(params))
    n_param = length(u_names)

    # All vars are approximately uniform in unconstrained space
    distributions = repeat([Parameterized(Normal(0.0, 1.0))], n_param)
    priors = ParameterDistribution(distributions, constraints, u_names)
    jldsave(joinpath(outdir_path, "prior.jld2"); distributions, constraints, u_names)
    return priors
end

""" Define reference simulations for loss function"""
function construct_reference_models()::Vector{ReferenceModel}
    les_root = "/groups/esm/zhaoyi/pycles_clima"
    scm_root = "/groups/esm/hervik/calibration/static_input"  # path to folder with `Output.<scm_name>.00000` files

    # Calibrate using reference data and options described by the ReferenceModel struct.
    ref_bomex = ReferenceModel(
        # Define variables considered in the loss function
        y_names = ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"],
        # Reference data specification
        les_root = les_root,
        les_name = "Bomex",
        les_suffix = "aug09",
        # Simulation case specification
        scm_root = scm_root,
        scm_name = "Bomex",
        # Define observation window (s)
        t_start = 4.0 * 3600,  # 4hrs
        t_end = 24.0 * 3600,  # 24hrs
    )
    # Make vector of reference models
    ref_models::Vector{ReferenceModel} = [ref_bomex]
    @assert all(isdir.([les_dir.(ref_models)... scm_dir.(ref_models)...]))

    return ref_models
end

"""
    generate_ekobj(u::Array{Float64, 2},
               ref_stats::ReferenceStatistics,
               algo)

Generates an EnsembleKalmanProcess from a parameter
vector, ReferenceStatistics and an algorithm.
"""
function generate_ekobj(u::Array{Float64, 2}, ref_stats::ReferenceStatistics, outdir_path::String, algo)
    ekp = serialize_struct(EnsembleKalmanProcess(u, ref_stats.y, ref_stats.Γ, algo))
    jldsave(ekobj_path(outdir_path, 1); ekp)
    return
end


function init_calibration(N_ens::Int, N_iter::Int, job_id::String)

    outdir_root = pwd()
    # Define preconditioning and regularization of inverse problem
    perform_PCA = false # Performs PCA on data
    normalize = true  # whether to normalize data by pooled variance
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    model_type::Symbol = :les  # :les or :scm
    # Flags for saving output data
    save_eki_data = true  # eki output
    save_ensemble_data = false  # .nc-files from each ensemble run
    algo = Inversion() # Sampler(vcat(get_mean(priors)...), get_cov(priors))
    Δt = 1.0 # Artificial time stepper of the EKI.

    # Define the parameters that we want to learn
    params = Dict(
        # entrainment parameters
        "entrainment_factor" => [bounded(0.05, 0.5)],
        "detrainment_factor" => [bounded(0.05, 0.5)],
    )
    ref_models = construct_reference_models()
    ref_stats = ReferenceStatistics(ref_models, model_type, perform_PCA, normalize, tikhonov_noise = 1e-3)

    algo_type = typeof(algo) == Sampler{Float64} ? "eks" : "eki"
    n_param = length(collect(keys(params)))
    d = length(ref_stats.y)
    outdir_path =
        joinpath(outdir_root, "results_$(algo_type)_dt$(Δt)_p$(n_param)_e$(N_ens)_i$(N_iter)_d$(d)_$(model_type)")
    println("Name of outdir path for this EKP is: $outdir_path")
    mkpath(outdir_path)
    open("$(job_id).txt", "w") do io
        write(io, "$(outdir_path)\n")
    end
    priors = construct_priors(params, outdir_path)
    # parameters are sampled in unconstrained space
    initial_params = construct_initial_ensemble(priors, N_ens, rng_seed = rand(1:1000))
    generate_ekobj(initial_params, ref_stats, outdir_path, algo)
    params_cons_i = transform_unconstrained_to_constrained(priors, initial_params)
    params = [c[:] for c in eachcol(params_cons_i)]
    versions = map(param -> generate_scm_input(param, get_name(priors), ref_models, ref_stats, outdir_path), params)

    # Store version identifiers for this ensemble in a common file
    open(joinpath(outdir_path, "versions_1.txt"), "w") do io
        for version in versions
            write(io, "$(version)\n")
        end
    end
end

s = ArgParseSettings()
@add_arg_table s begin
    "--n_ens"
    help = "Number of ensemble members."
    arg_type = Int
end
@add_arg_table s begin
    "--n_it"
    help = "Number of algorithm iterations."
    arg_type = Int
end
@add_arg_table s begin
    "--job_id"
    help = "Job identifier"
    arg_type = String
    default = "default_id"
end
parsed_args = parse_args(ARGS, s)

init_calibration(parsed_args["n_ens"], parsed_args["n_it"], parsed_args["job_id"])
