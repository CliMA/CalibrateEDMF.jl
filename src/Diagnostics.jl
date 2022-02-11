module Diagnostics

using NCDatasets
using Statistics
using LinearAlgebra
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
import EnsembleKalmanProcesses: construct_sigma_ensemble, construct_mean, construct_cov
include(joinpath("../ekp_experimental", "failsafe_inversion.jl"))

using ..ReferenceModels
using ..ReferenceStats
include("helper_funcs.jl")
const NC = NCDatasets

export io_dictionary_ensemble, io_dictionary_reference, io_dictionary_metrics
export io_dictionary_particle_state, io_dictionary_particle_eval
export io_dictionary_val_metrics, io_dictionary_val_particle_eval
export io_dictionary_val_reference, io_dictionary_prior

"""Reference diagnostics dictionary."""
function io_dictionary_reference()
    io_dict = Dict(
        "Gamma" => (; dims = ("out", "out"), group = "reference", type = Float64),
        "Gamma_full" => (; dims = ("out_full", "out_full"), group = "reference", type = Float64),
        "y" => (; dims = ("out",), group = "reference", type = Float64),
        "y_full" => (; dims = ("out_full",), group = "reference", type = Float64),
        "P_pca" => (; dims = ("out_full", "out"), group = "reference", type = Float64),
        "num_vars" => (; dims = ("config",), group = "reference", type = Int16),
        "var_dof" => (; dims = ("config",), group = "reference", type = Int16),
        "config_pca_dim" => (; dims = ("config",), group = "reference", type = Int16),
        "config_name" => (; dims = ("config",), group = "reference", type = String),
        "config_dz" => (; dims = ("config",), group = "reference", type = Float64),
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
    var_dof = Int.([size(P_pca, 1) for P_pca in ref_stats.pca_vec] ./ num_vars)
    config_pca_dim = [size(P_pca, 2) for P_pca in ref_stats.pca_vec]
    config_name = [
        rm.case_name == "LES_driven_SCM" ? join(split(basename(rm.y_dir), ".")[2:end], "_") : rm.case_name
        for rm in ref_models
    ]
    config_dz = [get_dz(rm.scm_dir) for rm in ref_models]
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
        "y" => Base.setindex(orig_dict["y"], ref_stats.y, :field),
        "y_full" => Base.setindex(orig_dict["y_full"], ref_stats.y_full, :field),
        "P_pca" => Base.setindex(orig_dict["P_pca"], P_pca_full, :field),
        "num_vars" => Base.setindex(orig_dict["num_vars"], num_vars, :field),
        "var_dof" => Base.setindex(orig_dict["var_dof"], var_dof, :field),
        "config_pca_dim" => Base.setindex(orig_dict["config_pca_dim"], config_pca_dim, :field),
        "config_name" => Base.setindex(orig_dict["config_name"], config_name, :field),
        "config_dz" => Base.setindex(orig_dict["config_dz"], config_dz, :field),
    )
    if write_full_stats
        io_dict["Gamma_full"] = Base.setindex(orig_dict["Gamma_full"], Array(ref_stats.Γ_full), :field)
    end
    if all([length(norm_vec) == length(ref_stats.norm_vec[1]) for norm_vec in ref_stats.norm_vec])
        io_dict["norm_factor"] = Base.setindex(orig_dict["norm_factor"], hcat(ref_stats.norm_vec...)', :field)
    end
    return io_dict
end

function io_dictionary_val_reference()
    io_dict = Dict(
        "Gamma_val" => (; dims = ("out_val", "out_val"), group = "reference", type = Float64),
        "Gamma_full_val" => (; dims = ("out_full_val", "out_full_val"), group = "reference", type = Float64),
        "y_val" => (; dims = ("out_val",), group = "reference", type = Float64),
        "y_full_val" => (; dims = ("out_full_val",), group = "reference", type = Float64),
        "P_pca_val" => (; dims = ("out_full_val", "out_val"), group = "reference", type = Float64),
        "num_vars_val" => (; dims = ("config_val",), group = "reference", type = Int16),
        "var_dof_val" => (; dims = ("config_val",), group = "reference", type = Int16),
        "config_pca_dim_val" => (; dims = ("config_val",), group = "reference", type = Int16),
        "config_name_val" => (; dims = ("config_val",), group = "reference", type = String),
        "config_dz_val" => (; dims = ("config_val",), group = "reference", type = Float64),
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
    var_dof = Int.([size(P_pca, 1) for P_pca in ref_stats.pca_vec] ./ num_vars)
    config_pca_dim = [size(P_pca, 2) for P_pca in ref_stats.pca_vec]
    config_name = [
        rm.case_name == "LES_driven_SCM" ? join(split(basename(rm.y_dir), ".")[2:end], "_") : rm.case_name
        for rm in ref_models
    ]
    config_dz = [get_dz(rm.scm_dir) for rm in ref_models]
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
        "y_val" => Base.setindex(orig_dict["y_val"], ref_stats.y, :field),
        "y_full_val" => Base.setindex(orig_dict["y_full_val"], ref_stats.y_full, :field),
        "P_pca_val" => Base.setindex(orig_dict["P_pca_val"], P_pca_full, :field),
        "num_vars_val" => Base.setindex(orig_dict["num_vars_val"], num_vars, :field),
        "var_dof_val" => Base.setindex(orig_dict["var_dof_val"], var_dof, :field),
        "config_pca_dim_val" => Base.setindex(orig_dict["config_pca_dim_val"], config_pca_dim, :field),
        "config_name_val" => Base.setindex(orig_dict["config_name_val"], config_name, :field),
        "config_dz_val" => Base.setindex(orig_dict["config_dz_val"], config_dz, :field),
    )
    if write_full_stats
        io_dict["Gamma_full_val"] = Base.setindex(orig_dict["Gamma_full_val"], Array(ref_stats.Γ_full), :field)
    end
    if all([length(norm_vec) == length(ref_stats.norm_vec[1]) for norm_vec in ref_stats.norm_vec])
        io_dict["norm_factor_val"] = Base.setindex(orig_dict["norm_factor_val"], hcat(ref_stats.norm_vec...)', :field)
    end
    return io_dict
end

"""
    io_dictionary_prior()

Parameter prior diagnostics dictionary.

Elements:
 - u_mean_prior :: Prior mean in unconstrained parameter space.
 - phi_mean_prior :: Prior mean in constrained parameter space.
 - u_var_prior :: Diagonal of the prior covariance in unconstrained space.
 - phi_low_unc_prior :: Lower uncertainty bound (μ-1σ_prior) of prior in constrained space.
 - phi_upp_unc_prior :: Upper uncertainty bound (μ+1σ_prior) of prior in constrained space.
 - phi_low_std_prior :: Lower standard bound (μ-1) of prior in constrained space. Useful
                        measure of minimum allowed values for bounded parameters.
 - phi_upp_std_prior :: Upper standard bound (μ+1) of prior in constrained space. Useful
                        measure of maximum allowed values for bounded parameters.
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
    ϕ_low_std = transform_unconstrained_to_constrained(priors, u_mean .- 1.0)
    ϕ_upp_std = transform_unconstrained_to_constrained(priors, u_mean .+ 1.0)

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

Scalar metrics dictionary.

Elements:
 - loss_mean_g :: (ḡ - y)'Γ_inv(ḡ - y).
 - mse_full_mean :: Ensemble mean of MSE(g_full, y_full).
 - mse_full_min :: Ensemble min of MSE(g_full, y_full).
 - mse_full_max :: Ensemble max of MSE(g_full, y_full).
 - mse_full_var :: Variance estimate of MSE(g_full, y_full), empirical (EKI/EKS) or quadrature (UKI).
 - mse_full_nn_mean :: MSE(g_full, y_full) of particle closest to the mean in parameter space. The
                       mean in parameter space is the solution to the particle-based inversion.
 - failures :: Number of particle failures per iteration. If the calibration is run with the "high_loss"
               failure handler, this diagnostic will not capture the failures.
 - nn_mean_index :: Particle index of the nearest neighbor to the ensemble mean in parameter space.
"""
function io_dictionary_metrics()
    io_dict = Dict(
        "loss_mean_g" => (; dims = ("iteration",), group = "metrics", type = Float64),
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
function io_dictionary_metrics(ekp::EnsembleKalmanProcess, mse_full::Vector{FT}) where {FT <: Real}
    orig_dict = io_dictionary_metrics()

    # Failure-safe variance
    mse_full_var = get_metric_var(ekp, mse_full)

    # Get failures
    failures = length(filter(isnan, mse_full))

    # Get mse at nearest_to_mean point
    nn_mean = get_mean_nearest_neighbor(ekp)
    mse_full_nn_mean = mse_full[nn_mean]

    # Filter NaNs for statistics
    mse_filt = filter(!isnan, mse_full)

    io_dict = Dict(
        "loss_mean_g" => Base.setindex(orig_dict["loss_mean_g"], get_error(ekp)[end], :field),
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

function io_dictionary_val_metrics()
    io_dict = Dict(
        "val_mse_full_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_min" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_max" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_var" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_nn_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
    )
    return io_dict
end
function io_dictionary_val_metrics(ekp::EnsembleKalmanProcess, mse_full::Vector{FT}) where {FT <: Real}
    orig_dict = io_dictionary_val_metrics()

    # Failure-safe variance
    mse_full_var = get_metric_var(ekp, mse_full)

    # Get mse at nearest_to_mean point
    nn_mean = get_mean_nearest_neighbor(ekp)
    mse_full_nn_mean = mse_full[nn_mean]

    # Filter NaNs for statistics
    mse_filt = filter(!isnan, mse_full)

    io_dict = Dict(
        "val_mse_full_mean" => Base.setindex(orig_dict["val_mse_full_mean"], mean(mse_filt), :field),
        "val_mse_full_min" => Base.setindex(orig_dict["val_mse_full_min"], minimum(mse_filt), :field),
        "val_mse_full_max" => Base.setindex(orig_dict["val_mse_full_max"], maximum(mse_filt), :field),
        "val_mse_full_var" => Base.setindex(orig_dict["val_mse_full_var"], mse_full_var, :field),
        "val_mse_full_nn_mean" => Base.setindex(orig_dict["val_mse_full_nn_mean"], mse_full_nn_mean, :field),
    )
    return io_dict
end

"""
    io_dictionary_particle_state()

Dictionary of particle-wise parameter diagnostics, not involving forward model evaluations.

Elements:
 - u   :: Parameters in unconstrained (inverse problem) space.
 - phi :: Parameters in constrained (physical) space.
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

Dictionary of particle-wise parameter diagnostics involving forward model evaluations.

Elements:
 - g :: Forward model evaluation in inverse problem space.
 - g_full :: Forward model evaluation in primitive output space,
    normalized using the pooled field covariance.
 - mse_full :: Particle-wise evaluation of MSE(g_full, y_full).
"""
function io_dictionary_particle_eval()
    io_dict = Dict(
        "g" => (; dims = ("particle", "out", "iteration"), group = "particle_diags", type = Float64),
        "g_full" => (; dims = ("particle", "out_full", "iteration"), group = "particle_diags", type = Float64),
        "mse_full" => (; dims = ("particle", "iteration"), group = "particle_diags", type = Float64),
    )
    return io_dict
end
function io_dictionary_particle_eval(
    ekp::EnsembleKalmanProcess,
    g_full::Matrix{FT},
    mse_full::Vector{FT},
    d::IT,
) where {FT <: Real, IT <: Integer}
    orig_dict = io_dictionary_particle_eval()
    io_dict = Dict(
        "g" => Base.setindex(orig_dict["g"], get_g_final(ekp)'[:, 1:d], :field),
        "g_full" => Base.setindex(orig_dict["g_full"], g_full', :field),
        "mse_full" => Base.setindex(orig_dict["mse_full"], mse_full, :field),
    )
    return io_dict
end
function io_dictionary_particle_eval(ekp::EnsembleKalmanProcess, mse_full::Vector{FT}) where {FT <: Real}
    orig_dict = io_dictionary_particle_eval()
    io_dict = Dict("mse_full" => Base.setindex(orig_dict["mse_full"], mse_full, :field))
    return io_dict
end

function io_dictionary_val_particle_eval()
    io_dict = Dict(
        "val_g" => (; dims = ("particle", "out_val", "iteration"), group = "particle_diags", type = Float64),
        "val_g_full" => (; dims = ("particle", "out_full_val", "iteration"), group = "particle_diags", type = Float64),
        "val_mse_full" => (; dims = ("particle", "iteration"), group = "particle_diags", type = Float64),
    )
    return io_dict
end
function io_dictionary_val_particle_eval(g::Matrix{FT}, g_full::Matrix{FT}, mse_full::Vector{FT}) where {FT <: Real}
    orig_dict = io_dictionary_val_particle_eval()
    io_dict = Dict(
        "val_g" => Base.setindex(orig_dict["val_g"], g', :field),
        "val_g_full" => Base.setindex(orig_dict["val_g_full"], g_full', :field),
        "val_mse_full" => Base.setindex(orig_dict["val_mse_full"], mse_full, :field),
    )
    return io_dict
end
function io_dictionary_val_particle_eval(mse_full::Vector{FT}) where {FT <: Real}
    orig_dict = io_dictionary_val_particle_eval()
    io_dict = Dict("val_mse_full" => Base.setindex(orig_dict["val_mse_full"], mse_full, :field))
    return io_dict
end

"""Ensemble diagnostics dictionary."""
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

Inputs:
 - ekp    :: The EnsembleKalmanProcess.
 - priors :: The priors defining transformations between constrained and
    unconstrained space.
Outputs:
 - The parameter covariance in constrained space.

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

function get_metric_var(ekp::EnsembleKalmanProcess, metric::Vector{FT}) where {FT <: Real}
    if isa(ekp.process, Unscented)
        if any(isnan.(metric))
            succ_ens = [i for i = 1:length(metric) if !isnan(metric[i])]
            metric_mean = construct_failsafe_mean(ekp, metric, succ_ens)
            return construct_failsafe_cov(ekp, metric, metric_mean, succ_ens)
        else
            metric_mean = construct_mean(ekp, metric)
            return construct_cov(ekp, metric, metric_mean)
        end
    else
        return var(filter(!isnan, metric))
    end
end

"""Returns the index of the nearest neighbor to the ensemble mean parameter"""
function get_mean_nearest_neighbor(ekp::EnsembleKalmanProcess)
    u = get_u_final(ekp)
    u_mean = mean(u, dims = 2)
    return argmin(vcat(sum((u .- u_mean) .^ 2, dims = 1)...))
end

end # module
