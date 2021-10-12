# Import modules to all processes

@everywhere using Pkg
@everywhere Pkg.activate("../..")
@everywhere using Distributions
@everywhere using StatsBase
@everywhere using LinearAlgebra
@everywhere using CalibrateEDMF
@everywhere using CalibrateEDMF.ReferenceModels
@everywhere using CalibrateEDMF.ReferenceStats
@everywhere using CalibrateEDMF.TurbulenceConvectionUtils
# Import EKP modules
@everywhere using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
@everywhere using EnsembleKalmanProcesses.Observations
@everywhere using EnsembleKalmanProcesses.ParameterDistributionStorage
@everywhere const src_dir = dirname(pathof(CalibrateEDMF))
@everywhere include(joinpath(src_dir, "helper_funcs.jl"))
@everywhere include(joinpath(src_dir, "scampy_utils.jl"))
using Base
using JLD2
using NPZ


# Define preconditioning and regularization of inverse problem
normalized = true # Variable normalization
perform_PCA = true # PCA on config covariance
cutoff_reg = true # Regularize above PCA cutoff
beta = 10.0 # Regularization hyperparameter
variance_loss = 1.0e-3 # PCA variance loss
noisy_obs = true # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
model_type = :les  # :les or :scm


"""Define parameters and their priors"""
function construct_priors()
    # Define the parameters that we want to learn
    params = Dict(
        # entrainment parameters
        "entrainment_factor" => [bounded(0.01, 0.3)],
        "detrainment_factor" => [bounded(0.01, 0.9)],
        "sorting_power" => [bounded(0.25, 4.0)],
        "tke_ed_coeff" => [bounded(0.01, 0.5)],
        "tke_diss_coeff" => [bounded(0.01, 0.5)],
        "pressure_normalmode_adv_coeff" => [bounded(0.0, 0.5)],
        "pressure_normalmode_buoy_coeff1" => [bounded(0.0, 0.5)],
        "pressure_normalmode_drag_coeff" => [bounded(5.0, 15.0)],
        "static_stab_coeff" => [bounded(0.1, 0.8)],
    )
    param_names = collect(keys(params))
    constraints = collect(values(params))
    n_param = length(param_names)

    # All vars are approximately uniform in unconstrained space
    prior_dist = repeat([Parameterized(Normal(0.0, 0.5))], n_param)
    return ParameterDistribution(prior_dist, constraints, param_names)
end


"""Define reference simulations for loss function."""
function construct_reference_models()::Vector{ReferenceModel}
    les_root = "/groups/esm/ilopezgo"
    scm_root = "./tc_inputs"  # path to folder with `Output.<scm_name>.00000` files
    sim_names = ["DYCOMS_RF01", "GABLS", "Bomex"]
    les_suffixes = ["may20", "iles128wCov", "may18"]
    # Define variables per flow configuration
    y_names = Array{String, 1}[]
    push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #DYCOMS_RF01
    push!(y_names, ["thetal_mean", "u_mean", "v_mean", "tke_mean"]) #GABLS
    push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #Bomex
    # Define observation window per flow configuration
    ti = [7200.0, 25200.0, 14400.0]
    tf = [14400.0, 32400.0, 21600.0]

    @assert (  # Each entry in these lists correspond to one simulation case
        length(sim_names) == length(les_suffixes) == length(y_names) == length(ti) == length(tf)
    )

    # Calibrate using reference data and options described by the ReferenceModel struct.
    ref_models = [
        ReferenceModel(
            # Define variables considered in the loss function
            y_names = vars,
            # Reference data specification
            les_dir = data_directory("/groups/esm/ilopezgo", sim_name, les_suffix),
            # Simulation case specification
            scm_dir = data_directory("./tc_inputs", sim_name, "00000"),
            scm_name = sim_name,
            t_start = t_start,
            t_end = t_end,
        ) for (sim_name, les_suffix, vars, t_start, t_end) in zip(sim_names, les_suffixes, y_names, ti, tf)
    ]
    @assert all(isdir.([les_dir.(ref_models)... scm_dir.(ref_models)...]))
    return ref_models
end


"""Perform the calibration using Kalman methods."""
function run_calibrate()
    priors = construct_priors()
    ref_models = construct_reference_models()
    outdir_root = pwd()

    C = sum([length(ref_model.t_start) for ref_model in ref_models])
    sim_num = length(ref_models)
    ref_stats = ReferenceStatistics(
        ref_models,
        model_type,
        perform_PCA,
        normalized,
        variance_loss = variance_loss,
        tikhonov_noise = beta,
        tikhonov_mode = "relative",
        dim_scaling = true,
    )
    d = pca_length(ref_stats)

    algo = Unscented(vcat(get_mean(priors)...), get_cov(priors), 1.0, 0) # Sampler(vcat(get_mean(priors)...), get_cov(priors)) # Inversion()
    N_ens = typeof(algo) == Unscented{Float64, Int64} ? 2 * length(get_name(priors)) + 1 : 50 # number of ensemble members
    N_iter = 10 # number of EKP iterations.
    Δt = 1.0 # follows scaling by batch size

    @info "NUMBER OF ENSEMBLE MEMBERS: " N_ens
    @info "NUMBER OF ITERATIONS: " N_iter
    deterministic_forward_map = noisy_obs ? true : false

    if typeof(algo) != Unscented{Float64, Int64}
        initial_params = construct_initial_ensemble(priors, N_ens, rng_seed = rand(1:1000))
        # Discard unstable parameter combinations, parallel
        #precondition_ensemble!(initial_params, priors, param_names, y_names, ti, tf=tf)
    end

    # UKI does not require sampling from the prior
    ekobj = typeof(algo) == Unscented{Float64, Int64} ? EnsembleKalmanProcess(ref_stats.y, ref_stats.Γ, algo) :
        EnsembleKalmanProcess(initial_params, ref_stats.y, ref_stats.Γ, algo)
    @everywhere g_(x::Vector{Float64}) = run_SCM(x, get_name($priors), $ref_models, $ref_stats)

    # Create output dir
    prefix = perform_PCA ? "results_pycles_PCA_" : "results_pycles_"
    prefix = cutoff_reg ? string(prefix, "creg", beta, "_") : prefix
    prefix = typeof(algo) == Sampler{Float64} ? string(prefix, "eks_") : prefix
    prefix = typeof(algo) == Unscented{Float64, Int64} ? string(prefix, "uki_") : prefix
    prefix = noisy_obs ? prefix : string(prefix, "nfo_")
    prefix = Δt ≈ 1 ? prefix : string(prefix, "dt", Δt, "_")
    outdir_path = string(prefix, "p", length(get_name(priors)), "_e", N_ens, "_i", N_iter, "_d", d)

    @info "Name of outdir path for this EKP, " outdir_path
    try
        mkdir(outdir_path)
    catch e
        @warn "Output directory already exists. Output may be overwritten."
    end

    # EKP iterations
    g_ens = zeros(N_ens, d)
    norm_err_list = []
    g_big_list = []
    Δt_scaled = Δt / C # Scale artificial timestep by batch size
    for i in 1:N_iter
        # Note that the parameters are transformed when used as input to SCAMPy
        params_cons_i = deepcopy(transform_unconstrained_to_constrained(priors, get_u_final(ekobj)))
        params = [row[:] for row in eachrow(params_cons_i')]
        @everywhere params = $params
        array_of_tuples = pmap(g_, params) # Outer dim is params iterator
        (sim_dirs_arr, g_ens_arr, g_ens_arr_pca) = ntuple(l -> getindex.(array_of_tuples, l), 3) # Outer dim is G̃, G 
        println("LENGTH OF G_ENS_ARR", length(g_ens_arr))
        println("LENGTH OF G_ENS_ARR_PCA", length(g_ens_arr_pca))
        println(string("\n\nEKP evaluation $i finished. Updating ensemble ...\n"))
        for j in 1:N_ens
            g_ens[j, :] = g_ens_arr_pca[j]
        end
        # Get normalized error
        push!(norm_err_list, compute_errors(g_ens_arr, ref_stats.y_full))
        push!(g_big_list, g_ens_arr)
        if typeof(algo) == Inversion
            update_ensemble!(
                ekobj,
                Array(g_ens'),
                Δt_new = Δt_scaled,
                deterministic_forward_map = deterministic_forward_map,
            )
        elseif typeof(algo) == Unscented{Float64, Int64}
            update_ensemble!(ekobj, Array(g_ens'))  # TODO: Add Δt_new=Δt_scaled to EKP master branch (from ilopezgp/EKP.jl)
        else
            update_ensemble!(ekobj, Array(g_ens'))
        end
        println("\nEnsemble updated. Saving results to file...\n")
        # Save EKP information to file
        save(
            joinpath(outdir_path, "ekp.jld2"),
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
            "norm_err",
            norm_err_list,
            "truth_mean_big",
            ref_stats.y_full,
            "g_big",
            g_big_list,
            "truth_cov_big",
            ref_stats.Γ_full,
            "P_pca",
            ref_stats.pca_vec,
            "pool_var",
            ref_stats.norm_vec,
        )
        # Convert to arrays
        phi_params = Array{Array{Float64, 2}, 1}(transform_unconstrained_to_constrained(priors, get_u(ekobj)))
        phi_params_arr = zeros(i + 1, length(get_name(priors)), N_ens)
        g_big_arr = zeros(i, N_ens, length(ref_stats.y_full))
        for (k, elem) in enumerate(phi_params)
            phi_params_arr[k, :, :] = elem
            if k < i + 1
                g_big_arr[k, :, :] = hcat(g_big_list[k]...)'
            end
        end
        norm_err_arr = hcat(norm_err_list...)' # N_iter, N_ens
        npzwrite(string(outdir_path, "/y_mean.npy"), ekobj.obs_mean)
        npzwrite(string(outdir_path, "/Gamma_y.npy"), ekobj.obs_noise_cov)
        npzwrite(string(outdir_path, "/y_mean_big.npy"), ref_stats.y_full)
        npzwrite(string(outdir_path, "/Gamma_y_big.npy"), ref_stats.Γ_full)
        npzwrite(string(outdir_path, "/phi_params.npy"), phi_params_arr)
        npzwrite(string(outdir_path, "/norm_err.npy"), norm_err_arr)
        npzwrite(string(outdir_path, "/g_big.npy"), g_big_arr)
        for (l, P_pca) in enumerate(ref_stats.pca_vec)
            if C ≈ sim_num
                npzwrite(string(outdir_path, "/P_pca_", ref_models[l].les_name, ".npy"), P_pca)
                npzwrite(string(outdir_path, "/pool_var_", ref_models[l].les_name, ".npy"), ref_stats.norm_vec[l])
            else
                npzwrite(string(outdir_path, "/P_pca_", l, ".npy"), P_pca)
                npzwrite(string(outdir_path, "/pool_var_", l, ".npy"), ref_stats.norm_vec[l])
            end
        end

    end

    # EKP results: Has the ensemble collapsed toward the truth?
    println("\nEKP ensemble mean at last stage (original space):")
    println(mean(transform_unconstrained_to_constrained(priors, get_u(ekobj)), dims = 1))

    println("\nEnsemble covariance det. 1st iteration, transformed space.")
    println(det(cov((get_u(ekobj, 1)), dims = 1)))
    println("\nEnsemble covariance det. last iteration, transformed space.")
    println(det(cov(get_u_final(ekobj), dims = 2)))
end

run_calibrate()
