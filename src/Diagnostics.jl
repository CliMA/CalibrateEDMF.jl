module Diagnostics

using NCDatasets
using Statistics
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
import EnsembleKalmanProcesses.EnsembleKalmanProcessModule: construct_sigma_ensemble
import EnsembleKalmanProcesses.EnsembleKalmanProcessModule: construct_mean, construct_cov

using ..ReferenceModels
using ..ReferenceStats
include("helper_funcs.jl")
const NC = NCDatasets

export io_dictionary_ensemble, io_dictionary_reference, io_dictionary_metrics
export io_dictionary_particle_state, io_dictionary_particle_eval
export io_dictionary_val_metrics, io_dictionary_val_particle_eval
export io_dictionary_val_reference

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
    config_dz = [get_dz(rm.y_dir) for rm in ref_models]
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
    config_dz = [get_dz(rm.y_dir) for rm in ref_models]
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
"""
function io_dictionary_metrics()
    io_dict = Dict(
        "loss_mean_g" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "mse_full_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "mse_full_min" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "mse_full_max" => (; dims = ("iteration",), group = "metrics", type = Float64),
    )
    return io_dict
end
function io_dictionary_metrics(ekp::EnsembleKalmanProcess, mse_full::Vector{FT}) where {FT <: Real}
    orig_dict = io_dictionary_metrics()
    io_dict = Dict(
        "loss_mean_g" => Base.setindex(orig_dict["loss_mean_g"], get_error(ekp)[end], :field),
        "mse_full_mean" => Base.setindex(orig_dict["mse_full_mean"], mean(mse_full), :field),
        "mse_full_min" => Base.setindex(orig_dict["mse_full_min"], minimum(mse_full), :field),
        "mse_full_max" => Base.setindex(orig_dict["mse_full_max"], maximum(mse_full), :field),
    )
    return io_dict
end

function io_dictionary_val_metrics()
    io_dict = Dict(
        "val_mse_full_mean" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_min" => (; dims = ("iteration",), group = "metrics", type = Float64),
        "val_mse_full_max" => (; dims = ("iteration",), group = "metrics", type = Float64),
    )
    return io_dict
end
function io_dictionary_val_metrics(mse_full::Vector{FT}) where {FT <: Real}
    orig_dict = io_dictionary_val_metrics()
    io_dict = Dict(
        "val_mse_full_mean" => Base.setindex(orig_dict["val_mse_full_mean"], mean(mse_full), :field),
        "val_mse_full_min" => Base.setindex(orig_dict["val_mse_full_min"], minimum(mse_full), :field),
        "val_mse_full_max" => Base.setindex(orig_dict["val_mse_full_max"], maximum(mse_full), :field),
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
    g_full::Array{FT, 2},
    mse_full::Vector{FT},
) where {FT <: Real}
    orig_dict = io_dictionary_particle_eval()
    io_dict = Dict(
        "g" => Base.setindex(orig_dict["g"], get_g_final(ekp)', :field),
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
function io_dictionary_val_particle_eval(g::Array{FT, 2}, g_full::Array{FT, 2}, mse_full::Vector{FT}) where {FT <: Real}
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
    )
    return io_dict
end
function io_dictionary_ensemble(ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    orig_dict = io_dictionary_ensemble()
    u_mean = get_u_mean(ekp)
    u_cov = get_u_cov(ekp)
    # The estimator of the mean is valid in unconstrained space, so we must transform the mean.
    ϕ_mean = transform_unconstrained_to_constrained(priors, u_mean)
    # The covariance of ϕ is not the transformed covariance, this is just a linear approximator.
    ϕ_cov = get_ϕ_cov(ekp, priors)
    io_dict = Dict(
        "u_mean" => Base.setindex(orig_dict["u_mean"], u_mean, :field),
        "phi_mean" => Base.setindex(orig_dict["phi_mean"], ϕ_mean, :field),
        "u_cov" => Base.setindex(orig_dict["u_cov"], u_cov, :field),
        "phi_cov" => Base.setindex(orig_dict["phi_cov"], ϕ_cov, :field),
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

end # module
