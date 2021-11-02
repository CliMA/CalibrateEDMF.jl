# This is an example on training the TurbulenceConvection.jl implementation
# of the EDMF scheme with data generated using PyCLES or TurbulenceConvection.jl
# (perfect model setting).
#
# This example is fully parallelized and can be run in the Caltech Central
# cluster with the included script. Parallelization of the calibration
# process is carried out by Julia's pmap() function.

# Import modules to all processes
@everywhere using Pkg
@everywhere Pkg.activate("../../..")
@everywhere using CalibrateEDMF
@everywhere using CalibrateEDMF.ReferenceModels
@everywhere using CalibrateEDMF.ReferenceStats
@everywhere using CalibrateEDMF.LESUtils
@everywhere using CalibrateEDMF.TurbulenceConvectionUtils
@everywhere using CalibrateEDMF.Pipeline
@everywhere const src_dir = dirname(pathof(CalibrateEDMF))
@everywhere include(joinpath(src_dir, "helper_funcs.jl"))
# Import EKP modules
@everywhere using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
@everywhere using EnsembleKalmanProcesses.Observations
@everywhere using EnsembleKalmanProcesses.ParameterDistributionStorage
# include(joinpath(@__DIR__, "../../../src/viz/ekp_plots.jl"))
using JLD2

# Include calibration config file to define problem
include(joinpath(dirname(pwd()), "config.jl"))

function run_calibrate(config; return_ekobj = false)

    N_iter = config["process"]["N_iter"]
    N_ens = config["process"]["N_ens"]
    algo = config["process"]["algorithm"]
    Δt = config["process"]["Δt"]

    perform_PCA = config["regularization"]["perform_PCA"]
    # For now this is not used, pmap does not work within if scopes
    apply_preconditioning = config["regularization"]["precondition"]

    save_eki_data = config["output"]["save_eki_data"]
    save_ensemble_data = config["output"]["save_ensemble_data"]

    init_dict = init_calibration(N_ens, N_iter, config, mode = "pmap")
    ekobj = init_dict["ekobj"]
    priors = init_dict["priors"]
    ref_models = init_dict["ref_models"]
    ref_stats = init_dict["ref_stats"]
    d = init_dict["d"]
    n_param = init_dict["n_param"]
    outdir_path = init_dict["outdir_path"]

    # Precondition prior
    @everywhere precondition_param(x::Vector{FT}) where {FT <: Real} = precondition(x, $priors, $ref_models, $ref_stats)
    precond_params = pmap(precondition_param, [c[:] for c in eachcol(get_u_final(ekobj))])
    ekobj = generate_ekp(hcat(precond_params...), ref_stats, algo, outdir_path = outdir_path)

    # Define caller function
    @everywhere g_(x::Vector{FT}) where {FT <: Real} = run_SCM(x, $priors.names, $ref_models, $ref_stats)
    # EKP iterations
    g_ens = zeros(N_ens, d)
    norm_err_list = []
    g_big_list = []
    for i in 1:N_iter
        # Parameters are transformed to constrained space when used as input to TurbulenceConvection.jl
        params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekobj))
        g_output_list = pmap(g_, [c[:] for c in eachcol(params_cons_i)]; retry_delays = zeros(5)) # Outer dim is params iterator
        (sim_dirs_arr, g_ens_arr, g_ens_arr_pca) = ntuple(l -> getindex.(g_output_list, l), 3) # Outer dim is G̃, G 
        @info "\n\nEKP evaluation $i finished. Updating ensemble ...\n"
        for j in 1:N_ens
            g_ens[j, :] = perform_PCA ? g_ens_arr_pca[j] : g_ens_arr[j]
        end

        # Get normalized error
        if typeof(algo) != Sampler{Float64}
            update_ensemble!(ekobj, Array(g_ens'), Δt_new = Δt)
        else
            update_ensemble!(ekobj, Array(g_ens'))
        end
        @info "\nEnsemble updated. Saving results to file...\n"

        # Get normalized error for full dimensionality output
        push!(norm_err_list, compute_errors(g_ens_arr, ref_stats.y_full))
        norm_err_arr = hcat(norm_err_list...)' # N_iter, N_ens
        # Store full dimensionality output
        push!(g_big_list, g_ens_arr)

        # Convert to arrays
        phi_params = Array{Array{Float64, 2}, 1}(transform_unconstrained_to_constrained(priors, get_u(ekobj)))
        phi_params_arr = zeros(i + 1, n_param, N_ens)
        g_big_arr = zeros(i, N_ens, full_length(ref_stats))
        for (k, elem) in enumerate(phi_params)
            phi_params_arr[k, :, :] = elem
            if k < i + 1
                g_big_arr[k, :, :] = hcat(g_big_list[k]...)'
            end
        end
        if save_eki_data
            # Save calibration process information to JLD2 file
            save(
                joinpath(outdir_path, "calibration_results_iter_$i.jld2"),
                "ekp_u",
                transform_unconstrained_to_constrained(priors, get_u(ekobj)),
                "ekp_g",
                get_g(ekobj),
                "truth_mean",
                ekobj.obs_mean,
                "truth_cov",
                ekobj.obs_noise_cov,
                "ekp_err",
                ekobj.err,
                "truth_mean_big",
                ref_stats.y_full,
                "truth_cov_big",
                ref_stats.Γ_full,
                "P_pca",
                ref_stats.pca_vec,
                "pool_var",
                ref_stats.norm_vec,
                "g_big",
                g_big_list,
                "g_big_arr",
                g_big_arr,
                "norm_err",
                norm_err_list,
                "norm_err_arr",
                norm_err_arr,
                "phi_params",
                phi_params_arr,
            )
        end

        # make ekp plots
        # make_ekp_plots(outdir_path, priors.names)

        if save_ensemble_data
            eki_iter_path = joinpath(outdir_path, "EKI_iter_$i")
            mkpath(eki_iter_path)
            save_full_ensemble_data(eki_iter_path, sim_dirs_arr, ref_models)
        end
    end
    # EKP results: Has the ensemble collapsed toward the truth?
    @info string(
        "\nEKP ensemble mean at last stage (original space): ",
        $"mean(transform_unconstrained_to_constrained(priors, get_u_final(ekobj)), dims = 2)",
    )

    if return_ekobj
        return ekobj, outdir_path
    end
end

### RUN SIMULATION ###
run_calibrate(get_config())
