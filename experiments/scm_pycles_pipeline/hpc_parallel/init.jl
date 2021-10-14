"""Initializes a SCM calibration process."""

using Distributions
using StatsBase
using LinearAlgebra
using CalibrateEDMF
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.TurbulenceConvectionUtils
const src_dir = dirname(pathof(CalibrateEDMF))
include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
using JLD2


""" Define reference simulations for loss function"""
function construct_reference_models()::Vector{ReferenceModel}

    ### Calibrate on field campaigns
    # Calibrate using reference data and options described by the ReferenceModel struct.
    ref_bomex = ReferenceModel(
        # Define variables considered in the loss function
        y_names = ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"],
        # Reference path specification
        les_dir = "/groups/esm/zhaoyi/pycles_clima/Output.Bomex.aug09",
        # Simulation path specification
        scm_dir = "/groups/esm/hervik/calibration/static_input/Output.Bomex.00000",
        # Simulation casename specification
        case_name = "Bomex",
        # Define observation window (s)
        t_start = 4.0 * 3600,  # 4hrs
        t_end = 24.0 * 3600,  # 24hrs
    )

    # Make vector of reference models
    ref_models::Vector{ReferenceModel} = [ref_bomex]

    #### Calibrate on LES Library
    # # available LES library simulations stored in LES_library dict
    # cfsite_numbers = [17] # for all cfsites us LES_library["cfsite_numbers"]
    # forcing_models = ["HadGEM2-A"]
    # months = [7]
    # experiments = ["amip"]

    # ref_models = [
    #     ReferenceModel(
    #         # Define variables considered in the loss function
    #         y_names = ["thetal_mean", "ql_mean", "qt_mean"],
    #         # Reference path specification
    #         les_dir = get_cfsite_les_dir(
    #             cfsite_number,
    #             forcing_model = forcing_model,
    #             month = month,
    #             experiment = experiment,
    #         ),
    #         # Simulation path specification
    #         scm_dir = string(
    #             "/groups/esm/cchristo/calibration/static_input/scm/Output.LES_driven_SCM.",
    #             generate_uuid(cfsite_number, forcing_model = forcing_model, month = month, experiment = experiment),
    #         ),
    #         # Simulation casename specification
    #         case_name = "LES_driven_SCM",
    #         # Define observation window (s)
    #         t_start = 4.0 * 3600,  # 4hrs
    #         t_end = 24.0 * 3600,  # 24hrs
    #     )
    #     for
    #     (cfsite_number, forcing_model, month, experiment) in zip(cfsite_numbers, forcing_models, months, experiments)
    # ]

    @assert all(isdir.([les_dir.(ref_models)...]))

    return ref_models
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
    # Flag for overwritting SCM input file
    overwrite_scm_file = false

    algo = Inversion() # Sampler(vcat(get_mean(priors)...), get_cov(priors))
    Δt = 1.0 # Artificial time stepper of the EKI.

    # Define the parameters that we want to learn
    params = Dict(
        # entrainment parameters
        "entrainment_factor" => [bounded(0.1, 0.5)],
        "detrainment_factor" => [bounded(0.1, 0.5)],
    )
    ref_models = construct_reference_models()

    # Create input scm stats and namelist file if files don't already exist
    run_SCM(ref_models, overwrite = overwrite_scm_file)

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
    priors = construct_priors(params, outdir_path = outdir_path)
    # parameters are sampled in unconstrained space
    initial_params = construct_initial_ensemble(priors, N_ens, rng_seed = rand(1:1000))
    generate_ekp(initial_params, ref_stats, algo, outdir_path = outdir_path)
    params_cons_i = transform_unconstrained_to_constrained(priors, initial_params)
    params = [c[:] for c in eachcol(params_cons_i)]
    versions = map(param -> generate_scm_input(param, get_name(priors), ref_models, ref_stats, outdir_path), params)

    # Store version identifiers for this ensemble in a common file
    write_versions(versions, 1, outdir_path = outdir_path)
end

s = ArgParseSettings()
@add_arg_table s begin
    "--n_ens"
    help = "Number of ensemble members."
    arg_type = Int
    "--n_it"
    help = "Number of algorithm iterations."
    arg_type = Int
    "--job_id"
    help = "Job identifier"
    arg_type = String
    default = "default_id"
end
parsed_args = parse_args(ARGS, s)

init_calibration(parsed_args["n_ens"], parsed_args["n_it"], parsed_args["job_id"])
