module NetCDFIO

using NCDatasets
using Statistics
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

using ..ReferenceModels
using ..ReferenceStats
using ..Diagnostics
include("helper_funcs.jl")
const NC = NCDatasets

export NetCDFIO_Diags
export open_files, close_files, write_iteration
export init_iteration_io, init_particle_diags, init_metrics, init_ensemble_diags
export init_val_diagnostics
export io_reference, io_diagnostics, io_val_diagnostics


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
        priors::ParameterDistribution,
        val_ref_stats::Union{ReferenceStatistics, Nothing} = nothing,
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
            batch_size = get_entry(config["reference"], "batch_size", length(ref_stats.pca_vec))
            batch_size = isnothing(batch_size) ? length(ref_stats.pca_vec) : batch_size

            particle = Array(1:N_ens)
            out = Array(1:d)
            out_full = Array(1:d_full)
            configuration = Array(1:C)
            param = priors.names

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
            NC.defDim(reference_grp, "batch_size", batch_size)

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

            if !isnothing(val_ref_stats)
                d_full_val = full_length(val_ref_stats)
                d_val = pca_length(val_ref_stats)
                out_val = Array(1:d_val)
                out_full_val = Array(1:d_full_val)
                NC.defDim(particle_grp, "out_full_val", d_full_val)
                NC.defVar(particle_grp, "out_full_val", out_full_val, ("out_full_val",))
                NC.defDim(particle_grp, "out_val", d_val)
                NC.defVar(particle_grp, "out_val", out_val, ("out_val",))
            end

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

function open_files(diags)
    diags.root_grp = NC.Dataset(diags.filepath, "a")
    diags.ensemble_grp = diags.root_grp.group["ensemble_diags"]
    diags.particle_grp = diags.root_grp.group["particle_diags"]
    diags.metric_grp = diags.root_grp.group["metrics"]
    vars = diags.vars

    # Hack to avoid https://github.com/Alexander-Barth/NCDatasets.jl/issues/135
    vars["ensemble_diags"] = Dict{String, Any}()
    for k in keys(diags.ensemble_grp)
        vars["ensemble_diags"][k] = diags.ensemble_grp[k]
    end
    vars["particle_diags"] = Dict{String, Any}()
    for k in keys(diags.particle_grp)
        vars["particle_diags"][k] = diags.particle_grp[k]
    end
    vars["metrics"] = Dict{String, Any}()
    for k in keys(diags.metric_grp)
        vars["metrics"][k] = diags.metric_grp[k]
    end
end

function close_files(diags::NetCDFIO_Diags)
    close(diags.root_grp)
end

"""Adds a given field to an existing NetCDF Dataset."""
function add_field(diags::NetCDFIO_Diags, var_name::String; dims, group, type)
    NC.Dataset(diags.filepath, "a") do root_grp
        grp = root_grp.group[group]
        new_var = NC.defVar(grp, var_name, type, dims)
    end
end

"""Writes current field `data` to an existing variable in a NetCDF Dataset."""
function write_current(diags::NetCDFIO_Diags, var_name::String, data; group)
    var = diags.vars[group][var_name]
    last_dim = length(size(var))
    last_dim_end = size(var, last_dim)
    selectdim(var, last_dim, last_dim_end) .= data
end

"""Writes all current fields in a given io_dict to an existing NetCDF Dataset."""
function write_current_dict(diags::NetCDFIO_Diags, io_dict)
    for var in keys(io_dict)
        write_current(diags, var, io_dict[var].field; group = io_dict[var].group)
    end
end

function write_ref(diags::NetCDFIO_Diags, var_name::String, data)
    NC.Dataset(diags.filepath, "a") do root_grp
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

function init_io_dict(diags::NetCDFIO_Diags, io_dict)
    for var in keys(io_dict)
        add_field(diags, var; dims = io_dict[var].dims, group = io_dict[var].group, type = io_dict[var].type)
    end
end

function init_ensemble_diags(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    io_dict = io_dictionary_ensemble(ekp, priors)
    init_io_dict(diags, io_dict)
    # Write initial ensemble_state to file.
    open_files(diags)
    write_current_dict(diags, io_dict)
    close_files(diags)
end

function init_particle_diags(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    io_dict = io_dictionary_particle_eval()
    init_io_dict(diags, io_dict)
    io_dict = io_dictionary_particle_state(ekp, priors)
    init_io_dict(diags, io_dict)
    # Write initial particle_state to file.
    open_files(diags)
    write_current_dict(diags, io_dict)
    close_files(diags)
end

function init_metrics(diags::NetCDFIO_Diags)
    io_dict = io_dictionary_metrics()
    init_io_dict(diags, io_dict)
end

function init_val_metrics(diags::NetCDFIO_Diags)
    io_dict = io_dictionary_val_metrics()
    init_io_dict(diags, io_dict)
end

function init_val_particle_diags(diags::NetCDFIO_Diags)
    io_dict = io_dictionary_val_particle_eval()
    init_io_dict(diags, io_dict)
end

function init_val_diagnostics(diags::NetCDFIO_Diags)
    init_val_metrics(diags)
    init_val_particle_diags(diags)
end

function io_metrics(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, mse_full::Vector{FT}) where {FT <: Real}
    io_dict = io_dictionary_metrics(ekp, mse_full)
    write_current_dict(diags, io_dict)
end

function io_val_metrics(diags::NetCDFIO_Diags, mse_full::Vector{FT}) where {FT <: Real}
    io_dict = io_dictionary_val_metrics(mse_full)
    write_current_dict(diags, io_dict)
end

function io_ensemble_diags(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    io_dict = io_dictionary_ensemble(ekp, priors)
    write_current_dict(diags, io_dict)
end

function io_val_particle_diags(
    diags::NetCDFIO_Diags,
    mse_full::Vector{FT},
    g::Union{Array{FT, 2}, Nothing} = nothing,
    g_full::Union{Array{FT, 2}, Nothing} = nothing,
) where {FT <: Real}
    # Write eval diagnostics to file
    if !isnothing(g_full)
        io_dict = io_dictionary_val_particle_eval(g, g_full, mse_full)
    else
        io_dict = io_dictionary_val_particle_eval(mse_full)
    end
    write_current_dict(diags, io_dict)
end

function io_particle_diags_eval(
    diags::NetCDFIO_Diags,
    ekp::EnsembleKalmanProcess,
    mse_full::Vector{FT},
    g_full::Union{Array{FT, 2}, Nothing} = nothing,
) where {FT <: Real}
    # Write eval diagnostics to file
    if !isnothing(g_full)
        io_dict = io_dictionary_particle_eval(ekp, g_full, mse_full)
    else
        io_dict = io_dictionary_particle_eval(ekp, mse_full)
    end
    write_current_dict(diags, io_dict)
end

function io_particle_diags_state(
    diags::NetCDFIO_Diags,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
) where {FT <: Real}
    io_dict = io_dictionary_particle_state(ekp, priors)
    write_current_dict(diags, io_dict)
end

function io_diagnostics(
    diags::NetCDFIO_Diags,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    mse_full::Vector{FT},
    g_full::Union{Array{FT, 2}, Nothing} = nothing,
) where {FT <: Real}
    open_files(diags)
    io_metrics(diags, ekp, mse_full)
    io_ensemble_diags(diags, ekp, priors)
    io_particle_diags_eval(diags, ekp, mse_full, g_full)
    write_iteration(diags)
    io_particle_diags_state(diags, ekp, priors)
    close_files(diags)
end

function io_val_diagnostics(
    diags::NetCDFIO_Diags,
    mse_full::Vector{FT},
    g::Union{Array{FT, 2}, Nothing} = nothing,
    g_full::Union{Array{FT, 2}, Nothing} = nothing,
) where {FT <: Real}
    open_files(diags)
    io_val_metrics(diags, mse_full)
    io_val_particle_diags(diags, mse_full, g, g_full)
    close_files(diags)
end

function init_iteration_io(diags::NetCDFIO_Diags)
    open_files(diags)
    ensemble_t = diags.ensemble_grp["iteration"]
    @inbounds ensemble_t[1] = 0
    particle_t = diags.particle_grp["iteration"]
    @inbounds particle_t[1] = 0
    metric_t = diags.metric_grp["iteration"]
    @inbounds metric_t[1] = 0
    close_files(diags)
end

function write_iteration(diags::NetCDFIO_Diags)
    ensemble_t = diags.ensemble_grp["iteration"]
    @inbounds ensemble_t[end + 1] = ensemble_t[end] + 1

    particle_t = diags.particle_grp["iteration"]
    @inbounds particle_t[end + 1] = particle_t[end] + 1

    metric_t = diags.metric_grp["iteration"]
    @inbounds metric_t[end + 1] = metric_t[end] + 1
end


end # module
