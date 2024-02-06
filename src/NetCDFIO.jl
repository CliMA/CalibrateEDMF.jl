module NetCDFIO

using NCDatasets
const NC = NCDatasets

using Statistics
using DocStringExtensions
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

using ..ReferenceModels
using ..ReferenceStats
using ..Diagnostics
using ..HelperFuncs

export NetCDFIO_Diags
export open_files, close_files, write_iteration
export init_iteration_io, init_particle_diags, init_metrics, init_ensemble_diags
export init_val_diagnostics
export io_prior, io_reference, io_diagnostics, io_val_diagnostics

"""
    NetCDFIO_Diags

Struct to work with NetCDFIO diagnostics data. Stores direct references to ensemble, particle and metric data.

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
mutable struct NetCDFIO_Diags
    "Reference to full NetCDF file"
    root_grp::NC.NCDataset{Nothing}
    "Reference to the `ensmeble_diags` group in the NetCDF file"
    ensemble_grp::NC.NCDataset{NC.NCDataset{Nothing}}
    "Reference to the `particle_diags` group in the NetCDF file"
    particle_grp::NC.NCDataset{NC.NCDataset{Nothing}}
    "Reference to the `metrics` group in the NetCDF file"
    metric_grp::NC.NCDataset{NC.NCDataset{Nothing}}
    "Reference to the `reference` group in the NetCDF file"
    reference_grp::NC.NCDataset{NC.NCDataset{Nothing}}
    "Path to directory where the NetCDF file is located"
    outdir_path::String
    "Path to the NetCDF file"
    filepath::String
    "Nested dictionary with references to stored variables within each of the ensemble, particle and metrics groups"
    vars::Dict{String, Any} # TODO: Refactor `vars`. See https://github.com/CliMA/TurbulenceConvection.jl/pull/1214
    function NetCDFIO_Diags(
        config::Dict{Any, Any},
        outdir_path::String,
        ref_stats::ReferenceStatistics,
        N_ens::Integer,
        priors::ParameterDistribution,
        val_ref_stats::Union{ReferenceStatistics, Nothing} = nothing,
    )

        # Initialize properties with valid type:
        tmp = tempname()
        root_grp = NC.Dataset(tmp, "c")
        NC.defGroup(root_grp, "ensemble_diags")
        NC.defGroup(root_grp, "particle_diags")
        NC.defGroup(root_grp, "metrics")
        NC.defGroup(root_grp, "reference")
        ensemble_grp = root_grp.group["ensemble_diags"]
        particle_grp = root_grp.group["particle_diags"]
        metric_grp = root_grp.group["metrics"]
        reference_grp = root_grp.group["reference"]
        close(root_grp)

        filepath = joinpath(outdir_path, "Diagnostics.nc")

        # Remove the NC file if it already exists.
        isfile(filepath) && rm(filepath; force = true)

        NC.Dataset(filepath, "c") do root_grp

            # Parameter dimension
            p = ndims(priors)
            # Output dimension for global problem: full and low-dim encoding
            d_full = full_length(ref_stats)
            d = pca_length(ref_stats)

            # Number of configurations
            C = length(ref_stats.pca_vec)

            # Max number of fields per configuration
            num_vars = [length(norm_scale) for norm_scale in ref_stats.norm_vec]
            f = maximum(num_vars)
            # Max number of variable degrees of freedom per configuration
            var_dof_max = maximum(ref_stats.zdof)

            # Number of configuration per batch
            batch_size = get_entry(config["reference"], "batch_size", length(ref_stats.pca_vec))
            batch_size = isnothing(batch_size) ? length(ref_stats.pca_vec) : batch_size

            particle = Array(1:N_ens)
            param = priors.name
            out_full = Array(1:d_full)
            out = Array(1:d)
            configuration = Array(1:C)
            field = Array(1:f)
            batch_index = Array(1:batch_size)
            dof = Array(1:var_dof_max)

            # Ensemble diagnostics (over all particles)
            ensemble_grp = NC.defGroup(root_grp, "ensemble_diags")
            NC.defDim(ensemble_grp, "param", p)
            NC.defVar(ensemble_grp, "param", param, ("param",))
            NC.defDim(ensemble_grp, "iteration", Inf)
            NC.defVar(ensemble_grp, "iteration", Int16, ("iteration",))

            # Parameter prior diagnostics
            prior_grp = NC.defGroup(root_grp, "prior")
            NC.defDim(prior_grp, "param", p)
            NC.defVar(prior_grp, "param", param, ("param",))

            # Reference model and stats diagnostics
            reference_grp = NC.defGroup(root_grp, "reference")
            NC.defDim(reference_grp, "out_full", d_full)
            NC.defVar(reference_grp, "out_full", out_full, ("out_full",))
            NC.defDim(reference_grp, "out", d)
            NC.defVar(reference_grp, "out", out, ("out",))
            NC.defDim(reference_grp, "config", C)
            NC.defVar(reference_grp, "config", configuration, ("config",))
            NC.defDim(reference_grp, "config_field", f)
            NC.defVar(reference_grp, "config_field", field, ("config_field",))
            NC.defDim(reference_grp, "dof", var_dof_max)
            NC.defVar(reference_grp, "dof", dof, ("dof",))

            augmented = get_entry(config["process"], "augmented", false)
            if augmented
                d = d + p
            end

            # Particle diagnostics
            particle_grp = NC.defGroup(root_grp, "particle_diags")
            NC.defDim(particle_grp, "particle", N_ens)
            NC.defVar(particle_grp, "particle", particle, ("particle",))
            NC.defDim(particle_grp, "out_full", d_full)
            NC.defVar(particle_grp, "out_full", out_full, ("out_full",))
            NC.defDim(particle_grp, "out_aug", d)
            NC.defVar(particle_grp, "out_aug", Array(1:d), ("out_aug",))
            NC.defDim(particle_grp, "param", p)
            NC.defVar(particle_grp, "param", param, ("param",))
            NC.defDim(particle_grp, "config", C)
            NC.defVar(particle_grp, "config", configuration, ("config",))
            NC.defDim(particle_grp, "batch_index", batch_size)
            NC.defVar(particle_grp, "batch_index", batch_index, ("batch_index",))
            NC.defDim(particle_grp, "config_field", f)
            NC.defVar(particle_grp, "config_field", field, ("config_field",))

            NC.defDim(particle_grp, "iteration", Inf)
            NC.defVar(particle_grp, "iteration", Int16, ("iteration",))


            if !isnothing(val_ref_stats)
                d_full_val = full_length(val_ref_stats)
                d_val = pca_length(val_ref_stats)
                C_val = length(val_ref_stats.pca_vec)

                # Max number of fields per configuration
                num_vars_val = [length(norm_vec) for norm_vec in val_ref_stats.norm_vec]
                f_val = maximum(num_vars_val)
                # Max number of variable degrees of freedom per configuration
                var_dof_max_val = maximum(val_ref_stats.zdof)

                batch_size_val = get_entry(config["validation"], "batch_size", length(val_ref_stats.pca_vec))
                batch_size_val = isnothing(batch_size_val) ? length(val_ref_stats.pca_vec) : batch_size_val

                out_full_val = Array(1:d_full_val)
                out_val = Array(1:d_val)
                configuration_val = Array(1:C_val)
                field_val = Array(1:f_val)
                batch_index_val = Array(1:batch_size_val)
                dof_val = Array(1:var_dof_max_val)

                NC.defDim(reference_grp, "out_full_val", d_full_val)
                NC.defVar(reference_grp, "out_full_val", out_full_val, ("out_full_val",))
                NC.defDim(reference_grp, "out_val", d_val)
                NC.defVar(reference_grp, "out_val", out_val, ("out_val",))
                NC.defDim(reference_grp, "config_val", C_val)
                NC.defVar(reference_grp, "config_val", configuration_val, ("config_val",))
                NC.defDim(reference_grp, "config_field_val", f_val)
                NC.defVar(reference_grp, "config_field_val", field_val, ("config_field_val",))
                NC.defDim(reference_grp, "dof_val", var_dof_max_val)
                NC.defVar(reference_grp, "dof_val", dof_val, ("dof_val",))

                if augmented
                    d_val = d_val + p
                end

                NC.defDim(particle_grp, "out_full_val", d_full_val)
                NC.defVar(particle_grp, "out_full_val", out_full_val, ("out_full_val",))
                NC.defDim(particle_grp, "out_aug_val", d_val)
                NC.defVar(particle_grp, "out_aug_val", Array(1:d_val), ("out_aug_val",))
                NC.defDim(particle_grp, "batch_index_val", batch_size_val)
                NC.defVar(particle_grp, "batch_index_val", batch_index_val, ("batch_index_val",))
                NC.defDim(particle_grp, "config_val", C_val)
                NC.defVar(particle_grp, "config_val", configuration_val, ("config_val",))

            end

            # Calibration metrics
            metric_grp = NC.defGroup(root_grp, "metrics")
            NC.defDim(metric_grp, "iteration", Inf)
            NC.defVar(metric_grp, "iteration", Int16, ("iteration",))
        end
        vars = Dict{String, Any}()
        return new(root_grp, ensemble_grp, particle_grp, metric_grp, reference_grp, outdir_path, filepath, vars)
    end

    function NetCDFIO_Diags(filepath::String)
        vars = Dict{String, Any}()
        diags = nothing
        NC.Dataset(filepath, "r") do root_grp
            ensemble_grp = root_grp.group["ensemble_diags"]
            particle_grp = root_grp.group["particle_diags"]
            metric_grp = root_grp.group["metrics"]
            reference_grp = root_grp.group["reference"]
            diags = new(root_grp, ensemble_grp, particle_grp, metric_grp, reference_grp, dirname(filepath), filepath, vars)
        end
        return diags
    end
end

"""
    open_files(diags::NetCDFIO_Diags)

Open the NetCDF file associated with `diags` and store its ensemble, particle and metric data in the struct.

See also [close_files](@ref).
"""
function open_files(diags::NetCDFIO_Diags)
    diags.root_grp = NC.Dataset(diags.filepath, "a")
    diags.ensemble_grp = diags.root_grp.group["ensemble_diags"]
    diags.particle_grp = diags.root_grp.group["particle_diags"]
    diags.metric_grp = diags.root_grp.group["metrics"]
    vars = diags.vars

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

"""
    close_files(diags::NetCDFIO_Diags)

Close the NetCDF file associated with `diags`.

See also [open_files](@ref).
"""
function close_files(diags::NetCDFIO_Diags)
    close(diags.root_grp)
end

"""
    add_field(diags::NetCDFIO_Diags, var_name; dims, group, type)

Add a field with some `var_name` and `type` to some `group` in an existing NetCDF Dataset. The dataset is assumed closed.
"""
function add_field(diags::NetCDFIO_Diags, var_name::String; dims, group, type)
    NC.Dataset(diags.filepath, "a") do root_grp
        grp = root_grp.group[group]
        new_var = NC.defVar(grp, var_name, type, dims)
    end
end

"""
    write_current(diags::NetCDFIO_Diags, group, var, data)

Write current field `data` to an existing variable in an open NetCDF Dataset.

Note that the last dimension of the variable is always assumed to be `iteration`, and the write operation writes to
the last available index of this dimension. See [`write_iteration`](@ref) to extend the `iteration` dimension.
"""
function write_current(diags::NetCDFIO_Diags, group::String, var::String, data::Number)
    diags.vars[group][var][end] = data
end
function write_current(diags::NetCDFIO_Diags, group::String, var::String, data::AbstractVector)
    diags.vars[group][var][:, end] = data
end
function write_current(diags::NetCDFIO_Diags, group::String, var::String, data::AbstractMatrix)
    diags.vars[group][var][:, :, end] = data
end

"""
    write_current_dict(diags::NetCDFIO_Diags, io_dict)

Write all fields in the io_dict to a NetCDF Dataset. The dataset is assumed to be open.
"""
function write_current_dict(diags::NetCDFIO_Diags, io_dict)
    for var in keys(io_dict)
        write_current(diags, io_dict[var].group, var, io_dict[var].field)
    end
end

function init_io_dict(diags::NetCDFIO_Diags, io_dict)
    for var in keys(io_dict)
        add_field(diags, var; dims = io_dict[var].dims, group = io_dict[var].group, type = io_dict[var].type)
    end
end

function io_prior(diags::NetCDFIO_Diags, prior::ParameterDistribution)
    io_dict = io_dictionary_prior(prior)
    for var in keys(io_dict)
        add_field(diags, var; dims = io_dict[var].dims, group = io_dict[var].group, type = io_dict[var].type)
        NC.Dataset(diags.filepath, "a") do root_grp
            prior_grp = root_grp.group["prior"]
            var_val = prior_grp[var]
            var_val .= io_dict[var].field
        end
    end
end

function write_ref(diags::NetCDFIO_Diags, var_name::String, data)
    NC.Dataset(diags.filepath, "a") do root_grp
        reference_grp = root_grp.group["reference"]
        var = reference_grp[var_name]
        var .= data
    end
end

"""
    io_reference(diags::NetCDFIO_Diags, ref_stats, ref_models, [write_full_stats = true])

Initialize and write reference data to the `reference` group in the NetCDF diagnostics file.

If `write_full_stats` is true, also write Γ_full to file.
"""
function io_reference(
    diags::NetCDFIO_Diags,
    ref_stats::ReferenceStatistics,
    ref_models::Vector{ReferenceModel},
    write_full_stats::Bool = true,
)
    io_dict = io_dictionary_reference(ref_stats, ref_models, write_full_stats)
    for var in keys(io_dict)
        add_field(diags, var; dims = io_dict[var].dims, group = io_dict[var].group, type = io_dict[var].type)
        write_ref(diags, var, io_dict[var].field)
    end
end

function io_val_reference(
    diags::NetCDFIO_Diags,
    ref_stats::ReferenceStatistics,
    ref_models::Vector{ReferenceModel},
    write_full_stats::Bool = true,
)
    io_dict = io_dictionary_val_reference(ref_stats, ref_models, write_full_stats)
    for var in keys(io_dict)
        add_field(diags, var; dims = io_dict[var].dims, group = io_dict[var].group, type = io_dict[var].type)
        write_ref(diags, var, io_dict[var].field)
    end
end

"""
    init_ensemble_diags(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Initialize and write initial ensemble state to the `ensemble_diags` group in the NetCDF diagnostics file.
"""
function init_ensemble_diags(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    io_dict = io_dictionary_ensemble(ekp, priors)
    init_io_dict(diags, io_dict)
    # Write initial ensemble_state to file.
    open_files(diags)
    write_current_dict(diags, io_dict)
    close_files(diags)
end

"""
    init_particle_diags(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Initialize and write initial particle state to the `particle` group in the NetCDF diagnostics file.
"""
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

"""
    init_metrics(diags::NetCDFIO_Diags)

Initialize the `metrics` group in the NetCDF diagnostics file.
"""
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

function init_val_diagnostics(
    diags::NetCDFIO_Diags,
    val_ref_stats::ReferenceStatistics,
    val_ref_models::Vector{ReferenceModel},
    write_full_stats::Bool = true,
)
    io_val_reference(diags, val_ref_stats, val_ref_models, write_full_stats)
    init_val_metrics(diags)
    init_val_particle_diags(diags)
end

"""
    io_metrics(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, mse_full::Vector)

Write current metrics data to the `metrics` group in the NetCDF diagnostics file.
"""
function io_metrics(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, mse_full::Vector{<:Real})
    io_dict = io_dictionary_metrics(ekp, mse_full)
    write_current_dict(diags, io_dict)
end

function io_val_metrics(
    diags::NetCDFIO_Diags,
    ekp::EnsembleKalmanProcess,
    val_ref_stats::ReferenceStatistics,
    g_val::Matrix{FT},
    val_mse_full::Vector{FT},
) where {FT <: Real}
    io_dict = io_dictionary_val_metrics(ekp, val_ref_stats, g_val, val_mse_full)
    write_current_dict(diags, io_dict)
end

"""
    io_ensemble_diags(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)

Write current ensemble diagnostics data to the `ensemble_diags` group in the NetCDF diagnostics file.
"""
function io_ensemble_diags(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    io_dict = io_dictionary_ensemble(ekp, priors)
    write_current_dict(diags, io_dict)
end

function io_val_particle_diags(
    diags::NetCDFIO_Diags,
    mse_full::Vector{FT},
    g::Matrix{FT},
    g_full::Matrix{FT},
    y_full::Vector{FT},
    batch_indices::Union{Vector{Int}, Nothing},
) where {FT <: Real}
    # Dimension of the outputs
    d_aug_val = length(diags.particle_grp["out_aug_val"])
    d_full_val = length(diags.particle_grp["out_full_val"])

    var_dofs = diags.reference_grp["var_dof"][:]
    norm_factors = diags.reference_grp["norm_factor"][:,:]
    var_names =  diags.reference_grp["ref_variable_names"][:,:]
    # If not minibatching - training set size
    batch_indices = isnothing(batch_indices) ? Array(diags.particle_grp["config_val"]) : batch_indices

    num_cases = length(batch_indices)
    # compute mse
    mse_by_var = compute_mse_by_var(g_full, y_full, var_dofs, var_names, norm_factors, num_cases)

    # Write eval diagnostics to file
    io_dict = io_dictionary_val_particle_eval(g, g_full, mse_full, mse_by_var, d_aug_val, d_full_val, batch_indices)
    write_current_dict(diags, io_dict)
end

function io_particle_diags_eval(
    diags::NetCDFIO_Diags,
    ekp::EnsembleKalmanProcess,
    mse_full::Vector{FT},
    g_full::Matrix{FT},
    y_full::Vector{FT},
    batch_indices::Union{Vector{Int}, Nothing},
) where {FT <: Real}
    # Dimension of the outputs
    d = length(diags.particle_grp["out_aug"])
    d_full = length(diags.particle_grp["out_full"])

    var_dofs = diags.reference_grp["var_dof"][:]
    norm_factors = diags.reference_grp["norm_factor"][:,:]
    var_names =  diags.reference_grp["ref_variable_names"][:,:]
    # If not minibatching - training set size
    batch_indices = isnothing(batch_indices) ? Array(diags.particle_grp["config"]) : batch_indices

    num_cases = length(batch_indices)
    # compute mse
    mse_by_var = compute_mse_by_var(g_full, y_full, var_dofs, var_names, norm_factors, num_cases)
    # Write eval diagnostics to file
    io_dict = io_dictionary_particle_eval(ekp, g_full, mse_full, mse_by_var, d, d_full, batch_indices)
    write_current_dict(diags, io_dict)
end

function io_particle_diags_state(diags::NetCDFIO_Diags, ekp::EnsembleKalmanProcess, priors::ParameterDistribution)
    io_dict = io_dictionary_particle_state(ekp, priors)
    write_current_dict(diags, io_dict)
end

function io_diagnostics(
    diags::NetCDFIO_Diags,
    ekp::EnsembleKalmanProcess,
    priors::ParameterDistribution,
    mse_full::Vector{FT},
    g_full::Matrix{FT},
    y_full::Vector{FT},
    batch_indices::Union{Vector{Int}, Nothing},
) where {FT <: Real}
    open_files(diags)
    # Eval diagnostics
    io_metrics(diags, ekp, mse_full)
    io_particle_diags_eval(diags, ekp, mse_full, g_full, y_full, batch_indices)
    write_iteration(diags)
    # State diagnostics
    io_particle_diags_state(diags, ekp, priors)
    io_ensemble_diags(diags, ekp, priors)
    close_files(diags)
end

function io_val_diagnostics(
    diags::NetCDFIO_Diags,
    ekp::EnsembleKalmanProcess,
    mse_full_val::Vector{FT},
    g_val::Matrix{FT},
    g_full_val::Matrix{FT},
    val_ref_stats::ReferenceStatistics,
    y_full::Vector{FT},
    val_batch_indices::Union{Vector{Int}, Nothing},
) where {FT <: Real}
    open_files(diags)
    io_val_metrics(diags, ekp, val_ref_stats, g_val, mse_full_val)
    io_val_particle_diags(diags, mse_full_val, g_val, g_full_val, y_full, val_batch_indices)
    close_files(diags)
end

"""
    init_iteration_io(diags::NetCDFIO_Diags)

Initialize iteration count at 0 in the NetCDF diagnostic file. File is assumed to be closed.
"""
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

"""
    write_iteration(diags::NetCDFIO_Diags)

Extend the iteration count by 1 in the NetCDF diagnostics file. File is assumed to open.
"""
function write_iteration(diags::NetCDFIO_Diags)
    ensemble_t = diags.ensemble_grp["iteration"]
    @inbounds ensemble_t[end + 1] = ensemble_t[end] + 1

    particle_t = diags.particle_grp["iteration"]
    @inbounds particle_t[end + 1] = particle_t[end] + 1

    metric_t = diags.metric_grp["iteration"]
    @inbounds metric_t[end + 1] = metric_t[end] + 1
end


end # module
