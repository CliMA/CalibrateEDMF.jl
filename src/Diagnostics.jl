module Diagnostics

using NCDatasets
using Statistics
using LinearAlgebra
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
import EnsembleKalmanProcesses: construct_sigma_ensemble, construct_mean, construct_cov
import EnsembleKalmanProcesses: construct_successful_mean, construct_successful_cov

using ..ReferenceModels
using ..ReferenceStats
using ..HelperFuncs
const NC = NCDatasets

export io_dictionary_ensemble, io_dictionary_reference, io_dictionary_metrics
export io_dictionary_particle_state, io_dictionary_particle_eval
export io_dictionary_val_metrics, io_dictionary_val_particle_eval
export io_dictionary_val_reference, io_dictionary_prior

"""
    io_dictionary_reference()
    io_dictionary_reference(
        ref_stats::ReferenceStatistics,
        ref_models::Vector{ReferenceModel},
        write_full_stats::Bool = true,
    )

Dictionary of diagnostics for the [`ReferenceModel`](@ref)s and [`ReferenceStatistics`](@ref) that define the inverse problem.

See also [`io_dictionary_val_reference`](@ref).

# Elements
- `Gamma`           :: Covariance matrix in the inverse problem latent space (regularized low-dimensional encoding).
- `Gamma_full`      :: Covariance matrix of normalized observed variables in full space (possibly ill-conditioned). 
    Only written to file if `write_full_stats` is true.
- `Gamma_full_diag` :: Diagonal of `Gamma_full`, useful when `Gamma_full` is not written to file.
- `y`               :: Observations in the inverse problem latent space (low-dimensional encoding).
- `y_full`          :: Normalized observations in full space.
- `P_pca`           :: PCA projection matrix from full space to low-dimensional latent space.
- `num_vars`        :: Maximum number of observed fields (not dimensions) per [`ReferenceModel`](@ref).
- `var_dof`         :: Maximum number of degrees of freedom of each field per [`ReferenceModel`](@ref).
- `config_pca_dim`  :: Dimensionality of the latent space associated with each [`ReferenceModel`](@ref).
- `config_name`     :: Name of each [`ReferenceModel`](@ref) used to construct the inverse problem.
- `config_z_obs`    :: Vertical locations of the observations of each [`ReferenceModel`](@ref).
- `norm_factor`     :: Pooled variance used to normalize each field of each [`ReferenceModel`](@ref).
"""
function io_dictionary_reference()
    io_dict = Dict(
        "Gamma" => (; dims = ("out", "out"), group = "reference", type = Float64),
        "Gamma_full" => (; dims = ("out_full", "out_full"), group = "reference", type = Float64),
        "Gamma_full_diag" => (; dims = ("out_full",), group = "reference", type = Float64),
        "y" => (; dims = ("out",), group = "reference", type = Float64),
        "y_full" => (; dims = ("out_full",), group = "reference", type = Float64),
        "P_pca" => (; dims = ("out_full", "out"), group = "reference", type = Float64),
        "num_vars" => (; dims = ("config",), group = "reference", type = Int16),
        "var_dof" => (; dims = ("config",), group = "reference", type = Int16),
        "config_pca_dim" => (; dims = ("config",), group = "reference", type = Int16),
        "config_name" => (; dims = ("config",), group = "reference", type = String),
        "config_z_obs" => (; dims = ("config", "dof"), group = "reference", type = Float64),
        "norm_factor" => (; dims = ("config", "config_field"), group = "reference", type = Float64),
    )
    return io_dict
end
function io_dictionary_reference(
    ref_stats::ReferenceStatistics,
    ref_models::Vector{ReferenceModel},
    write_full_stats::Bool = true,
)
    orig_dict = io_dictionary_reference()
    d_full = full_length(ref_stats)
    d = pca_length(ref_stats)
    num_vars = [length(norm_scale) for norm_scale in ref_stats.norm_vec]
    var_dof = ref_stats.zdof
    config_pca_dim = [size(P_pca, 2) for P_pca in ref_stats.pca_vec]
    config_name = [
        rm.case_name == "LES_driven_SCM" ? join(split(basename(rm.y_dir), ".")[2:end], "_") : rm.case_name for
        rm in ref_models
    ]

    config_z_obs = zeros(length(ref_models), maximum(var_dof))
    for (i, rm) in enumerate(ref_models)
        z_obs = get_z_obs(rm)
        config_z_obs[i, 1:length(z_obs)] = z_obs
    end

    P_pca_full = zeros(d_full, d)
    idx_row = 1
    idx_col = 1
    for P_pca in ref_stats.pca_vec
        rows, cols = size(P_pca)
        P_pca_full[idx_row:(idx_row + rows - 1), idx_col:(idx_col + cols - 1)] = P_pca
        idx_row += rows
        idx_col += cols
    end
    io_dict = Dict(
        "Gamma" => Base.setindex(orig_dict["Gamma"], ref_stats.Γ, :field),
        "Gamma_full_diag" => Base.setindex(orig_dict["Gamma_full_diag"], Array(diag(ref_stats.Γ_full)), :field),
        "y" => Base.setindex(orig_dict["y"], ref_stats.y, :field),
        "y_full" => Base.setindex(orig_dict["y_full"], ref_stats.y_full, :field),
        "P_pca" => Base.setindex(orig_dict["P_pca"], P_pca_full, :field),
        "num_vars" => Base.setindex(orig_dict["num_vars"], num_vars, :field),
        "var_dof" => Base.setindex(orig_dict["var_dof"], var_dof, :field),
        "config_pca_dim" => Base.setindex(orig_dict["config_pca_dim"], config_pca_dim, :field),
        "config_name" => Base.setindex(orig_dict["config_name"], config_name, :field),
        "config_z_obs" => Base.setindex(orig_dict["config_z_obs"], config_z_obs, :field),
    )
    max_num_fields = maximum([length(norm_vec) for norm_vec in ref_stats.norm_vec])
    norm_factor = zeros(length(ref_stats.norm_vec), max_num_fields)
    for (i, norm_vec) in enumerate(ref_stats.norm_vec)
        num_fields = length(norm_vec)
        norm_factor[i, 1:num_fields] = norm_vec
    end
    io_dict["norm_factor"] = Base.setindex(orig_dict["norm_factor"], norm_factor, :field)

    if write_full_stats
        io_dict["Gamma_full"] = Base.setindex(orig_dict["Gamma_full"], Array(ref_stats.Γ_full), :field)
    end
    return io_dict
end

"""
    io_dictionary_val_reference()
    io_dictionary_val_reference(
        ref_stats::ReferenceStatistics,
        ref_models::Vector{ReferenceModel},
        write_full_stats::Bool = true,
    )

Dictionary of diagnostics for the [`ReferenceModel`](@ref)s and [`ReferenceStatistics`](@ref) in the validation set.

See also [`io_dictionary_reference`](@ref).

# Elements
- `Gamma_val`           :: Covariance matrix in latent space, using the same truncation as for the training set.
- `Gamma_full_val`      :: Covariance matrix of normalized observed variables in full space. 
    Only written to file if `write_full_stats` is true.
- `Gamma_full_diag_val` :: Diagonal of `Gamma_full_val`, useful when `Gamma_full_val` is not written to file.
- `y_val`               :: Observations in latent space, for observed fields in the validation set.
- `y_full_val`          :: Normalized observations in full space, for the validation set.
- `P_pca_val`           :: PCA projection matrix from full space to low-dimensional latent space, for the validation set.
- `num_vars_val`        :: Maximum number of observed fields (not dimensions) per validation [`ReferenceModel`](@ref).
- `var_dof_val`         :: Maximum number of degrees of freedom of each field per validation [`ReferenceModel`](@ref).
- `config_pca_dim_val`  :: Dimensionality of the latent space associated with each validation [`ReferenceModel`](@ref).
- `config_name_val`     :: Name of each [`ReferenceModel`](@ref) in the validation set.
- `config_z_obs_val`    :: Vertical locations of the observations of each validation [`ReferenceModel`](@ref).
- `norm_factor_val`     :: Pooled variance used to normalize each field of each validation [`ReferenceModel`](@ref).
"""
function io_dictionary_val_reference()
    io_dict = Dict(
        "Gamma_val" => (; dims = ("out_val", "out_val"), group = "reference", type = Float64),
        "Gamma_full_val" => (; dims = ("out_full_val", "out_full_val"), group = "reference", type = Float64),
        "Gamma_full_diag_val" => (; dims = ("out_full_val",), group = "reference", type = Float64),
        "y_val" => (; dims = ("out_val",), group = "reference", type = Float64),
        "y_full_val" => (; dims = ("out_full_val",), group = "reference", type = Float64),
        "P_pca_val" => (; dims = ("out_full_val", "out_val"), group = "reference", type = Float64),
        "num_vars_val" => (; dims = ("config_val",), group = "reference", type = Int16),
        "var_dof_val" => (; dims = ("config_val",), group = "reference", type = Int16),
        "config_pca_dim_val" => (; dims = ("config_val",), group = "reference", type = Int16),
        "config_name_val" => (; dims = ("config_val",), group = "reference", type = String),
        "config_z_obs_val" => (; dims = ("config_val", "dof_val"), group = "reference", type = Float64),
        "norm_factor_val" => (; dims = ("config_val", "config_field_val"), group = "reference", type = Float64),
    )
    return io_dict
end
function io_dictionary_val_reference(
    ref_stats::ReferenceStatistics,
    ref_models::Vector{ReferenceModel},
    write_full_stats::Bool = true,
)
    orig_dict = io_dictionary_val_reference()
    d_full = full_length(ref_stats)
    d = pca_length(ref_stats)
    num_vars = [length(norm_scale) for norm_scale in ref_stats.norm_vec]
    var_dof = ref_stats.zdof
    config_pca_dim = [size(P_pca, 2) for P_pca in ref_stats.pca_vec]
    config_name = [
        rm.case_name == "LES_driven_SCM" ? join(split(basename(rm.y_dir), ".")[2:end], "_") : rm.case_name for
        rm in ref_models
    ]

    config_z_obs = zeros(length(ref_models), maximum(var_dof))
    for (i, rm) in enumerate(ref_models)
        z_obs = get_z_obs(rm)
        config_z_obs[i, 1:length(z_obs)] = z_obs
    end

    P_pca_full = zeros(d_full, d)
    idx_row = 1
    idx_col = 1
    for P_pca in ref_stats.pca_vec
        rows, cols = size(P_pca)
        P_pca_full[idx_row:(idx_row + rows - 1), idx_col:(idx_col + cols - 1)] = P_pca
        idx_row += rows
        idx_col += cols
    end
    io_dict = Dict(
        "Gamma_val" => Base.setindex(orig_dict["Gamma_val"], ref_stats.Γ, :field),
        "Gamma_full_diag_val" =>
            Base.setindex(orig_dict["Gamma_full_diag_val"], Array(diag(ref_stats.Γ_full)), :field),
        "y_val" => Base.setindex(orig_dict["y_val"], ref_stats.y, :field),
        "y_full_val" => Base.setindex(orig_dict["y_full_val"], ref_stats.y_full, :field),
        "P_pca_val" => Base.setindex(orig_dict["P_pca_val"], P_pca_full, :field),
        "num_vars_val" => Base.setindex(orig_dict["num_vars_val"], num_vars, :field),
        "var_dof_val" => Base.setindex(orig_dict["var_dof_val"], var_dof, :field),
        "config_pca_dim_val" => Base.setindex(orig_dict["config_pca_dim_val"], config_pca_dim, :field),
        "config_name_val" => Base.setindex(orig_dict["config_name_val"], config_name, :field),
        "config_z_obs_val" => Base.setindex(orig_dict["config_z_obs_val"], config_z_obs, :field),
    )
    max_num_fields = maximum([length(norm_vec) for norm_vec in ref_stats.norm_vec])
    norm_factor = zeros(length(ref_stats.norm_vec), max_num_fields)
    for (i, norm_vec) in enumerate(ref_stats.norm_vec)
        num_fields = length(norm_vec)
        norm_factor[i, 1:num_fields] = norm_vec
    end
    io_dict["norm_factor_val"] = Base.setindex(orig_dict["norm_factor_val"], norm_factor, :field)

    if write_full_stats
        io_dict["Gamma_full_val"] = Base.setindex(orig_dict["Gamma_full_val"], Array(ref_stats.Γ_full), :field)
    end
    return io_dict
end

"""
    io_dictionary_prior()
    io_dictionary_prior(priors::ParameterDistribution)

Parameter prior diagnostics dictionary.

# Elements
- `u_mean_prior`        :: Prior mean in unconstrained parameter space.
- `phi_mean_prior`      :: Prior mean in constrained parameter space.
- `u_var_prior`         :: Diagonal of the prior covariance in unconstrained space.
- `phi_low_unc_prior`   :: Lower uncertainty bound (μ-1σ_prior) of prior in constrained space.
- `phi_upp_unc_prior`   :: Upper uncertainty bound (μ+1σ_prior) of prior in constrained space.
- `phi_low_std_prior`   :: Lower standard bound (μ-1) of prior in constrained space. Useful measure of minimum allowed
    values for bounded parameters.
- `phi_upp_std_prior`   :: Upper standard bound (μ+1) of prior in constrained space. Useful measure of maximum allowed
    values for bounded parameters.
"""
function io_dictionary_prior()
    io_dict = Dict(
        "u_mean_prior" => (; dims = ("param",), group = "prior", type = Float64),
        "phi_mean_prior" => (; dims = ("param",), group = "prior", type = Float64),
        "u_var_prior" => (; dims = ("param",), group = "prior", type = Float64),
        "phi_low_unc_prior" => (; dims = ("param",), group = "prior", type = Float64),
        "phi_upp_unc_prior" => (; dims = ("param",), group = "prior", type = Float64),
        "phi_low_std_prior" => (; dims = ("param",), group = "prior", type = Float64),
        "phi_upp_std_prior" => (; dims = ("param",), group = "prior", type = Float64),
    )
    return io_dict
end
function io_dictionary_prior(priors::ParameterDistribution)
    orig_dict = io_dictionary_prior()
    u_mean = mean(priors)
    u_var = var(priors)
    # The estimator of the mean is valid in unconstrained space, so we must transform the mean.
    ϕ_mean = transform_unconstrained_to_constrained(priors, u_mean)
    # Transform prior uncertainty bands to constrained space
    u_low = u_mean .- sqrt.(u_var)
    u_upp = u_mean .+ sqrt.(u_var)
    ϕ_low = transform_unconstrained_to_constrained(priors, u_low)
    ϕ_upp = transform_unconstrained_to_constrained(priors, u_upp)
    # Transform standard uncertainty bands (1σ in unconstrained space) to constrained space
    ϕ_low_std = transform_unconstrained_to_constrained(priors, u_mean .- 1)
    ϕ_upp_std = transform_unconstrained_to_constrained(priors, u_mean .+ 1)

    io_dict = Dict(
        "u_mean_prior" => Base.setindex(orig_dict["u_mean_prior"], u_mean, :field),
        "phi_mean_prior" => Base.setindex(orig_dict["phi_mean_prior"], ϕ_mean, :field),
        "u_var_prior" => Base.setindex(orig_dict["u_var_prior"], u_var, :field),
        "phi_low_unc_prior" => Base.setindex(orig_dict["phi_low_unc_prior"], ϕ_low, :field),
        "phi_upp_unc_prior" => Base.setindex(orig_dict["phi_upp_unc_prior"], ϕ_upp, :field),
        "phi_low_std_prior" => Base.setindex(orig_dict["phi_low_std_prior"], ϕ_low_std, :field),
        "phi_upp_std_prior" => Base.setindex(orig_dict["phi_upp_std_prior"], ϕ_upp_std, :field),
    )
    return io_dict
end

"""
    io_dictionary_metrics()
    io_dictionary_metrics(ekp::EnsembleKalmanProcess, mse_full::Vector{<:Real})

Scalar metrics dictionary.

Evaluations of the data-model mismatch in inverse problem (i.e., latent) space are denoted `loss`. Errors computed in
normalized physical (i.e., full) space are denoted `mse_full`. Differences between these two metrics include:
- Covariance matrix defining the inner product (covariance weighting in `loss` vs L2 norm in `mse_full`),
- Treatment of trailing eigenvalues (truncation and regularization vs considering all eigenmodes).
- The `loss` includes the L2 penalty term, `mse_full` does not.

# Elements
- `loss_mean_g`         :: `(ḡ - y)'Γ_inv(ḡ - y)`. This is the ensemble mean loss seen by the Kalman inversion process.
- `loss_mean`           :: Ensemble mean of `(g - y)'Γ_inv(g - y)`.
- `loss_min`            :: Ensemble min of `(g - y)'Γ_inv(g - y)`.
- `loss_max`            :: Ensemble max of `(g - y)'Γ_inv(g - y)`.
- `loss_var`            :: Variance estimate of `(g - y)'Γ_inv(g - y)`, empirical (EKI/EKS) or quadrature (UKI).
- `loss_nn_mean`        :: `(g_nn - y)'Γ_inv(nn - y)`, where `g_nn` is the forward model output at the particle closest
    to the mean in parameter space.
- `mse_full_mean`       :: Ensemble mean of MSE(`g_full`, `y_full`).
- `mse_full_min`        :: Ensemble min of MSE(`g_full`, `y_full`).
- `mse_full_max`        :: Ensemble max of MSE(`g_full`, `y_full`).
- `mse_full_var`        :: Variance estimate of MSE(`g_full`, `y_full`), empirical (EKI/EKS) or quadrature (UKI).
- `mse_full_nn_mean`    :: MSE(`g_full`, `y_full`) of particle closest to the mean in parameter space. The mean in
    parameter space is the solution to the particle-based inversion.
- `failures`            :: Number of particle failures per iteration. If the calibration is run with the "high_loss"
    failure handler, this diagnostic will not capture the failures due to parameter mapping.
- `nn_mean_index`       :: Particle index of the nearest neighbor to the ensemble mean in parameter space. This index
    is used to construct `..._nn_mean` metrics.
"""
function io_dictionary_metrics()
    io_dict = Dict(
        "loss_mean_g" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "loss_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "loss_min" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "loss_max" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "loss_var" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "loss_nn_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "mse_full_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "mse_full_min" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "mse_full_max" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "mse_full_var" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "mse_full_nn_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "failures" => (; dims = ("iteration",), group = "metrics", type = Int16),
        "nn_mean_index" => (; dims = ("iteration",), group = "metrics", type = Int16),
    )
    return io_dict
end
function io_dictionary_metrics(ekp::EnsembleKalmanProcess{FT}, mse_full::Vector{FT}) where {FT <: Real}
    orig_dict = io_dictionary_metrics()

    # Get failures
    failures = length(filter(isnan, mse_full))

    # Get nearest_to_mean point
    nn_mean = get_mean_nearest_neighbor(ekp)

    # Failure-safe variance
    mse_full_var = get_metric_var(ekp, mse_full)
    # Get mse at nearest_to_mean point
    mse_full_nn_mean = mse_full[nn_mean]

    # Get loss (latent space)
    loss = compute_ensemble_loss(ekp)
    # Failure-safe variance
    loss_var = get_metric_var(ekp, loss)
    # Get loss at nearest_to_mean point
    loss_nn_mean = loss[nn_mean]

    # Filter NaNs
    loss_filt = filter(!isnan, loss)
    mse_filt = filter(!isnan, mse_full)

    io_dict = Dict(
        "loss_mean_g" => Base.setindex(orig_dict["loss_mean_g"], get_error(ekp)[end], :field),
        "loss_mean" => Base.setindex(orig_dict["loss_mean"], mean(loss_filt), :field),
        "loss_min" => Base.setindex(orig_dict["loss_min"], minimum(loss_filt), :field),
        "loss_max" => Base.setindex(orig_dict["loss_max"], maximum(loss_filt), :field),
        "loss_var" => Base.setindex(orig_dict["loss_var"], loss_var, :field),
        "loss_nn_mean" => Base.setindex(orig_dict["loss_nn_mean"], loss_nn_mean, :field),
        "mse_full_mean" => Base.setindex(orig_dict["mse_full_mean"], mean(mse_filt), :field),
        "mse_full_min" => Base.setindex(orig_dict["mse_full_min"], minimum(mse_filt), :field),
        "mse_full_max" => Base.setindex(orig_dict["mse_full_max"], maximum(mse_filt), :field),
        "mse_full_var" => Base.setindex(orig_dict["mse_full_var"], mse_full_var, :field),
        "mse_full_nn_mean" => Base.setindex(orig_dict["mse_full_nn_mean"], mse_full_nn_mean, :field),
        "failures" => Base.setindex(orig_dict["failures"], failures, :field),
        "nn_mean_index" => Base.setindex(orig_dict["nn_mean_index"], nn_mean, :field),
    )
    return io_dict
end

"""
    io_dictionary_val_metrics()
    io_dictionary_val_metrics(
        ekp::EnsembleKalmanProcess,
        val_ref_stats::ReferenceStatistics,
        g_val::Matrix,
        val_mse_full::Vector,
    )

Dictionary of scalar validation metrics.

Evaluations of the data-model mismatch in inverse problem (i.e., latent) space are denoted `loss`.
Errors computed in normalized physical (i.e., full) space are denoted `mse_full`. Differences between
these two metrics include:
- Covariance matrix defining the inner product (covariance weighting in `loss` vs L2 norm in `mse_full`),
- Treatment of trailing eigenvalues (truncation and regularization vs considering all eigenmodes).
- The `loss` includes the L2 penalty term, `mse_full` does not.
 
# Elements
- `val_loss_mean`     :: Ensemble mean of validation `(g - y)'Γ_inv(g - y)`.
- `val_loss_min`      :: Ensemble min of validation `(g - y)'Γ_inv(g - y)`.
- `val_loss_max`      :: Ensemble max of validation `(g - y)'Γ_inv(g - y)`.
- `val_loss_var`      :: Variance estimate of validation `(g - y)'Γ_inv(g - y)`, empirical (EKI/EKS) or quadrature (UKI).
- `val_loss_nn_mean`  :: Validation `(g_nn - y)'Γ_inv(nn - y)`, where `g_nn` is the validation forward model output at 
    the particle closest to the mean in parameter space.
- `val_mse_full_mean` :: Ensemble mean of MSE(`g_full_val`, `y_full_val`).
- `val_mse_full_min`  :: Ensemble min of MSE(`g_full_val`, `y_full_val`).
- `val_mse_full_max`  :: Ensemble max of MSE(`g_full_val`, `y_full_val`).
- `val_mse_full_var`  :: Variance estimate of MSE(`g_full_val`, `y_full_val`), empirical (EKI/EKS) or quadrature (UKI).
- `val_mse_full_nn_mean` :: MSE(`g_full_val`, `y_full_val`) of particle closest to the mean in parameter space.
    The mean in parameter space is the solution to the particle-based inversion.
"""
function io_dictionary_val_metrics()
    io_dict = Dict(
        "val_loss_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_loss_min" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_loss_max" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_loss_var" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_loss_nn_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_min" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_max" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_var" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_nn_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
    )
    return io_dict
end
function io_dictionary_val_metrics(
    ekp::EnsembleKalmanProcess{FT},
    val_ref_stats::ReferenceStatistics,
    g_val::Matrix{FT},
    val_mse_full::Vector{FT},
) where {FT <: Real}
    orig_dict = io_dictionary_val_metrics()

    # Get nearest_to_mean point
    nn_mean = get_mean_nearest_neighbor(ekp)

    # Failure-safe variance
    val_mse_full_var = get_metric_var(ekp, val_mse_full)
    # Get mse at nearest_to_mean point
    val_mse_full_nn_mean = val_mse_full[nn_mean]

    # Get loss (latent space), augmenting val_ref_stats if necessary
    d_aug = size(g_val, 1)
    d = length(val_ref_stats.y)
    if d_aug > d
        y_val = zeros(d_aug)
        y_val[1:d] = val_ref_stats.y
        y_val[(d + 1):d_aug] = ekp.obs_mean[(end - d_aug + d + 1):end]
        Γ_θ = ekp.obs_noise_cov[(end - d_aug + d + 1):end, (end - d_aug + d + 1):end]
        Γ_val = cat([val_ref_stats.Γ, Γ_θ]..., dims = (1, 2))
    else
        y_val = val_ref_stats.y
        Γ_val = val_ref_stats.Γ
    end

    val_loss = compute_ensemble_loss(g_val, y_val, Γ_val)
    # Failure-safe variance
    val_loss_var = get_metric_var(ekp, val_loss)
    # Get loss at nearest_to_mean point
    val_loss_nn_mean = val_loss[nn_mean]

    # Filter NaNs
    val_loss_filt = filter(!isnan, val_loss)
    val_mse_filt = filter(!isnan, val_mse_full)

    io_dict = Dict(
        "val_loss_mean" => Base.setindex(orig_dict["val_loss_mean"], mean(val_loss_filt), :field),
        "val_loss_min" => Base.setindex(orig_dict["val_loss_min"], minimum(val_loss_filt), :field),
        "val_loss_max" => Base.setindex(orig_dict["val_loss_max"], maximum(val_loss_filt), :field),
        "val_loss_var" => Base.setindex(orig_dict["val_loss_var"], val_loss_var, :field),
        "val_loss_nn_mean" => Base.setindex(orig_dict["val_loss_nn_mean"], val_loss_nn_mean, :field),
        "val_mse_full_mean" => Base.setindex(orig_dict["val_mse_full_mean"], mean(val_mse_filt), :field),
        "val_mse_full_min" => Base.setindex(orig_dict["val_mse_full_min"], minimum(val_mse_filt), :field),
        "val_mse_full_max" => Base.setindex(orig_dict["val_mse_full_max"], maximum(val_mse_filt), :field),
        "val_mse_full_var" => Base.setindex(orig_dict["val_mse_full_var"], val_mse_full_var, :field),
        "val_mse_full_nn_mean" => Base.setindex(orig_dict["val_mse_full_nn_mean"], val_mse_full_nn_mean, :field),
    )
    return io_dict
end

"""
    io_dictionary_particle_state()
    io_dictionary_particle_state(ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Dictionary of particle-wise parameter diagnostics, not involving forward model evaluations.

# Elements
- `u`   :: Parameter ensemble in unconstrained (inverse problem) space.
- `phi` :: Parameter ensemble in constrained (physical) space.
"""
function io_dictionary_particle_state()
    io_dict = Dict(
        "u" => (; dims = ("particle", "param", "iteration"), group = "particle_diags", type = Float64),
        "phi" => (; dims = ("particle", "param", "iteration"), group = "particle_diags", type = Float64),
    )
    return io_dict
end
function io_dictionary_particle_state(ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    orig_dict = io_dictionary_particle_state()
    u = get_u_final(ekp)
    ϕ = transform_unconstrained_to_constrained(priors, u)
    io_dict =
        Dict("u" => Base.setindex(orig_dict["u"], u', :field), "phi" => Base.setindex(orig_dict["phi"], ϕ', :field))
    return io_dict
end

"""
    io_dictionary_particle_eval()
    io_dictionary_particle_eval(
        ekp::EnsembleKalmanProcess{FT},
        g_full::Matrix{FT},
        mse_full::Vector{FT},
        d::IT,
        d_full::IT,
        batch_indices::Vector{IT},
    ) where {FT <: Real, IT <: Integer}

Dictionary of particle-wise diagnostics involving forward model evaluations.

# Elements
- `g`              :: Forward model evaluation in inverse problem space.
- `g_full`         :: Forward model evaluation in primitive output space, normalized using the pooled field covariance.
- `mse_full`       :: Particle-wise evaluation of MSE(`g_full`, `y_full`).
- `batch_indices`  :: Indices of [`ReferenceModel`](@ref)s evaluated per iteration.
"""
function io_dictionary_particle_eval()
    io_dict = Dict(
        "g" => (; dims = ("particle", "out_aug", "iteration"), group = "particle_diags", type = Float64),
        "g_full" => (; dims = ("particle", "out_full", "iteration"), group = "particle_diags", type = Float64),
        "mse_full" => (; dims = ("particle", "iteration"), group = "particle_diags", type = Float64),
        "batch_indices" => (; dims = ("batch_index", "iteration"), group = "particle_diags", type = Int16),
    )
    return io_dict
end
function io_dictionary_particle_eval(
    ekp::EnsembleKalmanProcess{FT},
    g_full::Matrix{FT},
    mse_full::Vector{FT},
    d::IT,
    d_full::IT,
    batch_indices::Vector{IT},
) where {FT <: Real, IT <: Integer}
    orig_dict = io_dictionary_particle_eval()

    g_aug = get_g_final(ekp)
    d_batch, N_ens = size(g_aug)
    # Fill "g" array with zeros and modify leading rows with possibly batched `g`
    g_filled = zeros(d, N_ens)
    g_filled[1:d_batch, :] = g_aug
    # Fill "g_full" array with zeros and modify leading rows with possibly batched `g`
    d_full_batch = size(g_full, 1)
    g_full_filled = zeros(d_full, N_ens)
    g_full_filled[1:d_full_batch, :] = g_full

    io_dict = Dict(
        "g" => Base.setindex(orig_dict["g"], g_filled', :field), # Avoid params in augmented state
        "g_full" => Base.setindex(orig_dict["g_full"], g_full_filled', :field),
        "mse_full" => Base.setindex(orig_dict["mse_full"], mse_full, :field),
        "batch_indices" => Base.setindex(orig_dict["batch_indices"], batch_indices, :field),
    )
    return io_dict
end

"""
    io_dictionary_val_particle_eval()
    io_dictionary_val_particle_eval(
        g::Matrix{FT},
        g_full::Matrix{FT},
        mse_full::Vector{FT},
        d::IT,
        d_full::IT,
        batch_indices::Vector{IT},
    ) where {FT <: Real, IT <: Integer}

Dictionary of particle-wise validation diagnostics involving forward model evaluations.

# Elements
- `val_g`              :: Validation forward model evaluation in reduced space.
- `val_g_full`         :: Validation forward model evaluation in primitive output space, 
normalized using the pooled field covariance.
- `val_mse_full`       :: Particle-wise evaluation of MSE(`val_g_full`, `val_y_full`).
- `val_batch_indices`  :: Indices of validation `ReferenceModel`s evaluated per iteration.
"""
function io_dictionary_val_particle_eval()
    io_dict = Dict(
        "val_g" => (; dims = ("particle", "out_aug_val", "iteration"), group = "particle_diags", type = Float64),
        "val_g_full" =>
            (; dims = ("particle", "out_full_val", "iteration"), group = "particle_diags", type = Float64),
        "val_mse_full" => (; dims = ("particle", "iteration"), group = "particle_diags", type = Float64),
        "val_batch_indices" => (; dims = ("batch_index_val", "iteration"), group = "particle_diags", type = Int16),
    )
    return io_dict
end
function io_dictionary_val_particle_eval(
    g::Matrix{FT},
    g_full::Matrix{FT},
    mse_full::Vector{FT},
    d_aug::IT,
    d_full::IT,
    batch_indices::Vector{IT},
) where {FT <: Real, IT <: Integer}
    orig_dict = io_dictionary_val_particle_eval()

    d_batch, N_ens = size(g)
    # Fill "g" array with zeros and modify leading rows with possibly batched `g`
    g_filled = zeros(d_aug, N_ens)
    g_filled[1:d_batch, :] = g
    # Fill "g_full" array with zeros and modify leading rows with possibly batched `g`
    d_full_batch = size(g_full, 1)
    g_full_filled = zeros(d_full, N_ens)
    g_full_filled[1:d_full_batch, :] = g_full

    io_dict = Dict(
        "val_g" => Base.setindex(orig_dict["val_g"], g_filled', :field),
        "val_g_full" => Base.setindex(orig_dict["val_g_full"], g_full_filled', :field),
        "val_mse_full" => Base.setindex(orig_dict["val_mse_full"], mse_full, :field),
        "val_batch_indices" => Base.setindex(orig_dict["val_batch_indices"], batch_indices, :field),
    )
    return io_dict
end

"""
    io_dictionary_ensemble()
    io_dictionary_ensemble(ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Dictionary of ensemble parameter diagnostics.

# Elements
- `u_mean`         :: Ensemble mean parameter in unconstrained (inverse problem) space.
- `phi_mean`       :: Ensemble mean parameter in constrained (physical) space.
- `u_cov`          :: Sample parameter covariance in unconstrained (inverse problem) space.
- `phi_cov`        :: Sample parameter covariance in constrained (physical) space.
- `phi_low_unc`    :: Lower uncertainty bound (μ-1σ) of the parameter value in constrained (physical) space.
- `phi_upp_unc`    :: Upper uncertainty bound (μ+1σ) of the parameter value in constrained (physical) space.
"""
function io_dictionary_ensemble()
    io_dict = Dict(
        "u_mean" => (; dims = ("param", "iteration"), group = "ensemble_diags", type = Float64),
        "phi_mean" => (; dims = ("param", "iteration"), group = "ensemble_diags", type = Float64),
        "u_cov" => (; dims = ("param", "param", "iteration"), group = "ensemble_diags", type = Float64),
        "phi_cov" => (; dims = ("param", "param", "iteration"), group = "ensemble_diags", type = Float64),
        "phi_low_unc" => (; dims = ("param", "iteration"), group = "ensemble_diags", type = Float64),
        "phi_upp_unc" => (; dims = ("param", "iteration"), group = "ensemble_diags", type = Float64),
    )
    return io_dict
end
function io_dictionary_ensemble(ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    orig_dict = io_dictionary_ensemble()
    u_mean = get_u_mean(ekp)
    u_cov = get_u_cov(ekp)
    # The estimator of the mean is valid in unconstrained space, so we must transform the mean.
    ϕ_mean = transform_unconstrained_to_constrained(priors, u_mean)
    # Transform uncertainty bands to constrained space
    u_low = u_mean .- sqrt.(diag(u_cov))
    u_upp = u_mean .+ sqrt.(diag(u_cov))
    ϕ_low = transform_unconstrained_to_constrained(priors, u_low)
    ϕ_upp = transform_unconstrained_to_constrained(priors, u_upp)
    # The covariance of ϕ is not the transformed covariance, this is just a linear approximator.
    ϕ_cov = get_ϕ_cov(ekp, priors)
    io_dict = Dict(
        "u_mean" => Base.setindex(orig_dict["u_mean"], u_mean, :field),
        "phi_mean" => Base.setindex(orig_dict["phi_mean"], ϕ_mean, :field),
        "u_cov" => Base.setindex(orig_dict["u_cov"], u_cov, :field),
        "phi_cov" => Base.setindex(orig_dict["phi_cov"], ϕ_cov, :field),
        "phi_low_unc" => Base.setindex(orig_dict["phi_low_unc"], ϕ_low, :field),
        "phi_upp_unc" => Base.setindex(orig_dict["phi_upp_unc"], ϕ_upp, :field),
    )
    return io_dict
end

function get_u_mean(ekp::EnsembleKalmanProcess)
    if isa(ekp.process, Unscented)
        return get_u_mean_final(ekp)
    else
        u = get_u_final(ekp)
        return vcat(mean(u, dims = 2)...)
    end
end

function get_u_cov(ekp::EnsembleKalmanProcess)
    if isa(ekp.process, Unscented)
        return deepcopy(ekp.process.uu_cov[end])
    else
        u = get_u_final(ekp)
        return cov(u, dims = 2)
    end
end

"""
    get_ϕ_cov(ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Get the last parameter covariance estimate in constrained (physical) space.

For ensemble methods, the covariance of the transformed parameters is returned.
For unscented methods, the covariance is computed through a quadrature on the
transformed quadrature points. The covariance of the transformed parameters
returned here is equal to the transformed covariance only under a first order
Taylor approximation, which is consistent with other approximations underlying the
calibration method.

# Arguments
- `ekp`    :: The [`EnsembleKalmanProcess`](@ref).
- `priors` :: The [`ParameterDistribution`](@ref) priors defining transformations between constrained and unconstrained space.

# Returns
- `Matrix` :: The parameter covariance in constrained space.

"""
function get_ϕ_cov(ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    if isa(ekp.process, Unscented)
        u_mean = get_u_mean_final(ekp)
        u_cov = deepcopy(ekp.process.uu_cov[end])
        u_p = construct_sigma_ensemble(ekp.process, u_mean, u_cov)
        ϕ_p = transform_unconstrained_to_constrained(priors, u_p)
        ϕ_p_mean = construct_mean(ekp, ϕ_p)
        return construct_cov(ekp, ϕ_p, ϕ_p_mean)
    else
        u = get_u_final(ekp)
        ϕ = transform_unconstrained_to_constrained(priors, u)
        return cov(ϕ, dims = 2)
    end
end

"""
    get_metric_var(ekp::EnsembleKalmanProcess, metric::Vector)

Compute the ensemble variance of a scalar metric.

For ensemble methods, the sample variance of the metric is returned. For unscented methods,
the variance is computed through a quadrature. Ensemble members where the metric is `NaN`
are filtered out of the computation.

# Arguments
- `ekp`     :: The EnsembleKalmanProcess.
- `metric`  :: A vector containing the value of the metric for each ensemble member.

# Returns
- `Real`    :: The ensemble variance of `metric`.

"""
function get_metric_var(ekp::EnsembleKalmanProcess{FT}, metric::Vector{FT}) where {FT <: Real}
    if isa(ekp.process, Unscented)
        if any(isnan.(metric))
            succ_ens = [i for i = 1:length(metric) if !isnan(metric[i])]
            metric_mean = construct_successful_mean(ekp, metric, succ_ens)
            return construct_successful_cov(ekp, metric, metric_mean, succ_ens)
        else
            metric_mean = construct_mean(ekp, metric)
            return construct_cov(ekp, metric, metric_mean)
        end
    else
        return var(filter(!isnan, metric))
    end
end

"""
    get_mean_nearest_neighbor(ekp::EnsembleKalmanProcess)

Return the index of the nearest neighbor to the ensemble mean parameter, in unconstrained space.
"""
function get_mean_nearest_neighbor(ekp::EnsembleKalmanProcess)
    u = get_u_final(ekp)
    u_mean = mean(u, dims = 2)
    return argmin(vcat(sum((u .- u_mean) .^ 2, dims = 1)...))
end

"""
    compute_ensemble_loss(g::Matrix, y::Vector, Γ::Matrix)
    compute_ensemble_loss(ekp::EnsembleKalmanProcess)

Compute the covariance-weighted error `(g - y)'Γ_inv(g - y)` for each ensemble member.
"""
function compute_ensemble_loss(
    g::AbstractMatrix{FT},
    y::AbstractVector{FT},
    Γ::Union{AbstractMatrix{FT}, UniformScaling{FT}},
) where {FT <: Real}
    diff = g .- y # [d, N_ens]
    loss = zeros(size(g, 2)) # [N_ens, 1]
    for i in 1:size(g, 2)
        Γ_inv_diff = Γ \ diff[:, i] # [d, 1]
        loss[i] = dot(diff[:, i], Γ_inv_diff)
    end
    return loss
end
function compute_ensemble_loss(ekp::EnsembleKalmanProcess)
    g = get_g_final(ekp)
    return compute_ensemble_loss(g, ekp.obs_mean, ekp.obs_noise_cov)
end

end # module
