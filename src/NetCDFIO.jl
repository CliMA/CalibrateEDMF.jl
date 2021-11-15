module NetCDFIO

using NCDatasets
using Statistics
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

using ..ReferenceModels
using ..ReferenceStats
include("helper_funcs.jl")
const NC = NCDatasets

export NetCDFIO_Diags
export open_files, close_files, write_iteration
export init_iteration_io, io_reference, init_particle_diags, io_particle_diags
export init_metrics, io_metrics


mutable struct NetCDFIO_Diags
    root_grp::NC.NCDataset{Nothing}
    ensemble_grp::NC.NCDataset{NC.NCDataset{Nothing}}
    particle_grp::NC.NCDataset{NC.NCDataset{Nothing}}
    metric_grp::NC.NCDataset{NC.NCDataset{Nothing}}
    outdir_path::String
    filepath::String
    vars::Dict{String, Any} # Hack to avoid https://github.com/Alexander-Barth/NCDatasets.jl/issues/135
    function NetCDFIO_Diags(
        config::Dict{Any, Any},
        outdir_path::String,
        ref_stats::ReferenceStatistics,
        ekp::EnsembleKalmanProcess,
    )

        # Initialize properties with valid type:
        tmp = tempname()
        root_grp = NC.Dataset(tmp, "c")
        NC.defGroup(root_grp, "ensemble_diags")
        NC.defGroup(root_grp, "particle_diags")
        NC.defGroup(root_grp, "metrics")
        ensemble_grp = root_grp.group["ensemble_diags"]
        particle_grp = root_grp.group["particle_diags"]
        metric_grp = root_grp.group["metrics"]
        close(root_grp)

        filepath = joinpath(outdir_path, "Diagnostics.nc")

        # Remove the NC file if it already exists.
        isfile(filepath) && rm(filepath; force = true)

        NC.Dataset(filepath, "c") do root_grp

            # Fetch dimensionality
            p, N_ens = size(get_u_final(ekp))
            d_full = full_length(ref_stats)
            d = pca_length(ref_stats)
            C = length(ref_stats.pca_vec)

            particle = Array(1:N_ens)
            out = Array(1:d)
            out_full = Array(1:d_full)
            param = Array(1:p)
            configuration = Array(1:C)

            # Ensemble diagnostics (over all particles)
            ensemble_grp = NC.defGroup(root_grp, "ensemble_diags")
            NC.defDim(ensemble_grp, "out_full", d_full)
            NC.defVar(ensemble_grp, "out_full", out_full, ("out_full",))
            NC.defDim(ensemble_grp, "out", d)
            NC.defVar(ensemble_grp, "out", out, ("out",))
            NC.defDim(ensemble_grp, "param", p)
            NC.defVar(ensemble_grp, "param", param, ("param",))
            NC.defDim(ensemble_grp, "iteration", Inf)
            NC.defVar(ensemble_grp, "iteration", Int16, ("iteration",))

            # Reference model and stats diagnostics
            reference_grp = NC.defGroup(root_grp, "reference")
            NC.defDim(reference_grp, "out_full", d_full)
            NC.defVar(reference_grp, "out_full", out_full, ("out_full",))
            NC.defDim(reference_grp, "out", d)
            NC.defVar(reference_grp, "out", out, ("out",))
            NC.defDim(reference_grp, "config", C)
            NC.defVar(reference_grp, "config", configuration, ("config",))

            # Particle diagnostics
            particle_grp = NC.defGroup(root_grp, "particle_diags")
            NC.defDim(particle_grp, "particle", N_ens)
            NC.defVar(particle_grp, "particle", particle, ("particle",))
            NC.defDim(particle_grp, "out_full", d_full)
            NC.defVar(particle_grp, "out_full", out_full, ("out_full",))
            NC.defDim(particle_grp, "out", d)
            NC.defVar(particle_grp, "out", out, ("out",))
            NC.defDim(particle_grp, "param", p)
            NC.defVar(particle_grp, "param", param, ("param",))
            NC.defDim(particle_grp, "iteration", Inf)
            NC.defVar(particle_grp, "iteration", Int16, ("iteration",))

            # Calibration metrics
            metric_grp = NC.defGroup(root_grp, "metrics")
            NC.defDim(metric_grp, "iteration", Inf)
            NC.defVar(metric_grp, "iteration", Int16, ("iteration",))
        end
        vars = Dict{String, Any}()
        return new(root_grp, ensemble_grp, particle_grp, metric_grp, outdir_path, filepath, vars)
    end

    function NetCDFIO_Diags(filepath::String)
        vars = Dict{String, Any}()
        diags = nothing
        NC.Dataset(filepath, "r") do root_grp
            ensemble_grp = root_grp.group["ensemble_diags"]
            particle_grp = root_grp.group["particle_diags"]
            metric_grp = root_grp.group["metrics"]
            diags = new(root_grp, ensemble_grp, particle_grp, metric_grp, dirname(filepath), filepath, vars)
        end
        return diags
    end
end

# IO Dictionaries
function io_dictionary_reference()
    io_dict = Dict(
        "Gamma" => (; dims = ("out", "out"), group = "reference", type = Float64),
        "Gamma_full" => (; dims = ("out_full", "out_full"), group = "reference", type = Float64),
        "y" => (; dims = ("out",), group = "reference", type = Float64),
        "y_full" => (; dims = ("out_full",), group = "reference", type = Float64),
        "num_vars" => (; dims = ("config",), group = "reference", type = Int16),
        "var_dof" => (; dims = ("config",), group = "reference", type = Int16),
        "config_pca_dim" => (; dims = ("config",), group = "reference", type = Int16),
        "config_name" => (; dims = ("config",), group = "reference", type = String),
        "config_dz" => (; dims = ("config",), group = "reference", type = Float64),
    )
    return io_dict
end
function io_dictionary_reference(ref_stats::ReferenceStatistics, ref_models::Vector{ReferenceModel})
    orig_dict = io_dictionary_reference()
    num_vars = [length(norm_scale) for norm_scale in ref_stats.norm_vec]
    var_dof = Int.([size(P_pca, 1) for P_pca in ref_stats.pca_vec] ./ num_vars)
    config_pca_dim = [size(P_pca, 2) for P_pca in ref_stats.pca_vec]
    config_name = [
        rm.case_name == "LES_driven_SCM" ? join(split(basename(rm.y_dir), ".")[2:end], "_") : rm.case_name
        for rm in ref_models
    ]
    config_dz = [get_dz(rm.y_dir) for rm in ref_models]
    io_dict = Dict(
        "Gamma" => Base.setindex(orig_dict["Gamma"], ref_stats.Γ, :field),
        "Gamma_full" => Base.setindex(orig_dict["Gamma_full"], ref_stats.Γ_full, :field),
        "y" => Base.setindex(orig_dict["y"], ref_stats.y, :field),
        "y_full" => Base.setindex(orig_dict["y_full"], ref_stats.y_full, :field),
        "num_vars" => Base.setindex(orig_dict["num_vars"], num_vars, :field),
        "var_dof" => Base.setindex(orig_dict["var_dof"], var_dof, :field),
        "config_pca_dim" => Base.setindex(orig_dict["config_pca_dim"], config_pca_dim, :field),
        "config_name" => Base.setindex(orig_dict["config_name"], config_name, :field),
        "config_dz" => Base.setindex(orig_dict["config_dz"], config_dz, :field),
    )
    return io_dict
end

"""
    io_dictionary_metrics()

Metrics dictionary

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

Particle evaluation diagnostics dictionary

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

function open_files(self)
    self.root_grp = NC.Dataset(self.filepath, "a")
    self.ensemble_grp = self.root_grp.group["ensemble_diags"]
    self.particle_grp = self.root_grp.group["particle_diags"]
    self.metric_grp = self.root_grp.group["metrics"]
    vars = self.vars

    # Hack to avoid https://github.com/Alexander-Barth/NCDatasets.jl/issues/135
    vars["ensemble_diags"] = Dict{String, Any}()
    for k in keys(self.ensemble_grp)
        vars["ensemble_diags"][k] = self.ensemble_grp[k]
    end
    vars["particle_diags"] = Dict{String, Any}()
    for k in keys(self.particle_grp)
        vars["particle_diags"][k] = self.particle_grp[k]
    end
    vars["metrics"] = Dict{String, Any}()
    for k in keys(self.metric_grp)
        vars["metrics"][k] = self.metric_grp[k]
    end
end

function close_files(self::NetCDFIO_Diags)
    close(self.root_grp)
end


function add_field(self::NetCDFIO_Diags, var_name::String; dims, group, type)
    NC.Dataset(self.filepath, "a") do root_grp
        grp = root_grp.group[group]
        new_var = NC.defVar(grp, var_name, type, dims)
    end
end

function write_current(self::NetCDFIO_Diags, var_name::String, data; group)
    var = self.vars[group][var_name]
    last_dim = length(size(var))
    last_dim_end = size(var, last_dim)
    selectdim(var, last_dim, last_dim_end) .= data
end

function write_ref(self::NetCDFIO_Diags, var_name::String, data)
    NC.Dataset(self.filepath, "a") do root_grp
        reference_grp = root_grp.group["reference"]
        var = reference_grp[var_name]
        var .= data
    end
end

function io_reference(diags::NetCDFIO_Diags, ref_stats::ReferenceStatistics, ref_models::Vector{ReferenceModel})
    io_dict = io_dictionary_reference(ref_stats, ref_models)
    for var in keys(io_dict)
        add_field(diags, var; dims = io_dict[var].dims, group = io_dict[var].group, type = io_dict[var].type)
        write_ref(diags, var, io_dict[var].field)
    end
end

function init_particle_diags(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    io_dict = io_dictionary_particle_eval()
    for var in keys(io_dict)
        add_field(diags, var; dims = io_dict[var].dims, group = io_dict[var].group, type = io_dict[var].type)
    end
    io_dict = io_dictionary_particle_state(ekp, priors)
    for var in keys(io_dict)
        add_field(diags, var; dims = io_dict[var].dims, group = io_dict[var].group, type = io_dict[var].type)
    end
    open_files(diags)
    for var in keys(io_dict)
        write_current(diags, var, io_dict[var].field; group = io_dict[var].group)
    end
    close_files(diags)
end

function init_metrics(diags::NetCDFIO_Diags)
    io_dict = io_dictionary_metrics()
    for var in keys(io_dict)
        add_field(diags, var; dims = io_dict[var].dims, group = io_dict[var].group, type = io_dict[var].type)
    end
end

function io_metrics(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, mse_full::Vector{FT}) where {FT <: Real}
    io_dict = io_dictionary_metrics(ekp, mse_full)
    for var in keys(io_dict)
        write_current(diags, var, io_dict[var].field; group = io_dict[var].group)
    end
end

function io_particle_diags(
    diags::NetCDFIO_Diags,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    g_full::Array{FT, 2},
    mse_full::Vector{FT},
) where {FT <: Real}
    # Write eval diagnostics to file
    io_dict = io_dictionary_particle_eval(ekp, g_full, mse_full)
    for var in keys(io_dict)
        write_current(diags, var, io_dict[var].field; group = io_dict[var].group)
    end
    # Write state diagnostics of next iteration to file
    write_iteration(diags)
    io_dict = io_dictionary_particle_state(ekp, priors)
    for var in keys(io_dict)
        write_current(diags, var, io_dict[var].field; group = io_dict[var].group)
    end
end

function init_iteration_io(self::NetCDFIO_Diags)
    ensemble_t = self.ensemble_grp["iteration"]
    @inbounds ensemble_t[1] = 0
    particle_t = self.particle_grp["iteration"]
    @inbounds particle_t[1] = 0
    metric_t = self.metric_grp["iteration"]
    @inbounds metric_t[1] = 0
end

function write_iteration(self::NetCDFIO_Diags)
    ensemble_t = self.ensemble_grp["iteration"]
    @inbounds ensemble_t[end + 1] = ensemble_t[end] + 1

    particle_t = self.particle_grp["iteration"]
    @inbounds particle_t[end + 1] = particle_t[end] + 1

    metric_t = self.metric_grp["iteration"]
    @inbounds metric_t[end + 1] = metric_t[end] + 1
end


end # module
