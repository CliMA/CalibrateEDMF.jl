module TurbulenceConvectionUtils

using JLD2
using JSON
using Random
using ..ReferenceModels
using ..ReferenceStats
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributionStorage
import EnsembleKalmanProcesses.EnsembleKalmanProcessModule: construct_initial_ensemble
using TurbulenceConvection
include(joinpath(@__DIR__, "helper_funcs.jl"))

export ModelEvaluator
export run_SCM, run_SCM_handler, get_scm_namelist, run_reference_SCM
export generate_scm_input, get_gcm_les_uuid, eval_single_ref_model
export save_full_ensemble_data
export precondition

"""
    struct ModelEvaluator
    
A structure containing the information required to perform
a forward model evaluation at a given parameter vector.
"""
Base.@kwdef struct ModelEvaluator{FT <: Real}
    "Parameter vector in constrained (i.e. physical) space"
    param_cons::Vector{FT}
    "Parameter names associated with parameter vector"
    param_names::Vector{String}
    "Vector of reference models"
    ref_models::Vector{ReferenceModel}
    "Reference statistics for the inverse problem"
    ref_stats::ReferenceStatistics

    function ModelEvaluator(
        param_cons::Vector{FT},
        param_names::Vector{String},
        RM::Vector{ReferenceModel},
        RS::ReferenceStatistics,
    ) where {FT <: Real}
        return new{FT}(param_cons, param_names, RM, RS)
    end
end


"""
    run_SCM(
        u::Vector{FT},
        u_names::Vector{String},
        RM::Vector{ReferenceModel},
        RS::ReferenceStatistics;
        error_check::Bool = false,
        namelist_args = nothing,
        failure_handler = "high_loss",
    ) where {FT <: Real}

Run the single-column model (SCM) using a set of parameters u 
and return the value of outputs defined in y_names, possibly 
after normalization and projection onto lower dimensional 
space using PCA.

Inputs:
 - u               :: Values of parameters to be used in simulations.
 - u_names         :: SCM names for parameters `u`.
 - RM              :: Vector of `ReferenceModel`s
 - RS              :: reference statistics for simulation
 - error_check     :: Returns as an additional argument whether the SCM call errored.
 - namelist_args   :: Additional arguments passed to the TurbulenceConvection namelist.
 - failure_handler :: Method used to handle failed simulations.
Outputs:
 - sim_dirs    :: Vector of simulation output directories
 - g_scm       :: Vector of model evaluations concatenated for all flow configurations.
 - g_scm_pca   :: Projection of `g_scm` onto principal subspace spanned by eigenvectors.
 - model_error :: Whether the simulation errored with the requested configuration.
"""
function run_SCM(
    u::Vector{FT},
    u_names::Vector{String},
    RM::Vector{ReferenceModel},
    RS::ReferenceStatistics;
    error_check::Bool = false,
    namelist_args = nothing,
    failure_handler = "high_loss",
) where {FT <: Real}

    mkpath(joinpath(pwd(), "tmp"))
    result_arr = map(x -> eval_single_ref_model(x..., RS, u, u_names, namelist_args), enumerate(RM))
    # Unpack
    sim_dirs, g_scm, g_scm_pca, sim_errors = (getindex.(result_arr, i) for i in 1:4)
    g_scm = vcat(g_scm...)
    g_scm_pca = vcat(g_scm_pca...)

    model_error = any(sim_errors)
    if model_error && failure_handler == "high_loss"
        g_scm .= 1e5
        g_scm_pca .= 1e5
    end

    @info "Length of g_scm (full): $(length(g_scm))"
    @info "Length of g_scm (pca) : $(length(g_scm_pca))"
    if error_check
        return sim_dirs, g_scm, g_scm_pca, model_error
    else
        return sim_dirs, g_scm, g_scm_pca
    end
end
function run_SCM(
    ME::ModelEvaluator;
    error_check::Bool = false,
    namelist_args = nothing,
    failure_handler = "high_loss",
) where {FT <: Real}
    return run_SCM(
        ME.param_cons,
        ME.param_names,
        ME.ref_models,
        ME.ref_stats,
        error_check = error_check,
        namelist_args = namelist_args,
        failure_handler = failure_handler,
    )
end

"""
    eval_single_ref_model(
        m_index::IT,
        m::ReferenceModel,
        RS::ReferenceStatistics,
        u::Vector{FT},
        u_names::Vector{String},
        namelist_args = nothing,
    ) where {FT <: Real, IT <: Int}

Runs the single-column model (SCM) under a single configuration
(i.e., ReferenceModel) using a set of parameters u, and returns
the forward model evaluation in both the original and the latent
PCA space.

Inputs:
 - m_index       :: The index of the ReferenceModel within the overarching
                    ref_models vector used to construct the ReferenceStatistics.
 - m             :: A ReferenceModel.
 - RS            :: reference statistics for simulation
 - u             :: Values of parameters to be used in simulations.
 - u_names       :: SCM names for parameters `u`.
 - namelist_args :: Additional arguments passed to the TurbulenceConvection namelist.
Outputs:
 - sim_dir     ::  Simulation output directory.
 - g_scm       :: Forward model evaluation in original output space.
 - g_scm_pca   :: Projection of `g_scm` onto principal subspace spanned by eigenvectors.
 - model_error :: Whether the simulation errored with the requested configuration.
"""
function eval_single_ref_model(
    m_index::IT,
    m::ReferenceModel,
    RS::ReferenceStatistics,
    u::Vector{FT},
    u_names::Vector{String},
    namelist_args = nothing,
) where {FT <: Real, IT <: Int}
    # create temporary directory to store SCM data in
    tmpdir = mktempdir(joinpath(pwd(), "tmp"))
    # run TurbulenceConvection.jl. Get output directory for simulation data
    sim_dir, model_error = run_SCM_handler(m, tmpdir, u, u_names, namelist_args)
    if model_error
        g_scm = fill(NaN, length(get_height(sim_dir)) * length(m.y_names))
    else
        g_scm = get_profile(m, sim_dir, z_scm = get_height(sim_dir))
        g_scm = normalize_profile(g_scm, length(m.y_names), RS.norm_vec[m_index])
    end
    # perform PCA reduction
    g_scm_pca = RS.pca_vec[m_index]' * g_scm
    return sim_dir, g_scm, g_scm_pca, model_error
end

"""
    get_scm_namelist(m::ReferenceModel; overwrite::Bool = false)::Dict

Fetch the namelist stored in `scm_dir(m)`.
Generate a new namelist if it doesn't exist or `overwrite=true`.
"""
function get_scm_namelist(m::ReferenceModel; overwrite::Bool = false)::Dict
    namelist_path = namelist_directory(scm_dir(m), m)
    namelist = if ~isfile(namelist_path) | overwrite
        NameList.default_namelist(m.case_name, root = scm_dir(m))
    else
        JSON.parsefile(namelist_path)
    end
    return namelist
end

"""
    run_reference_SCM(m::ReferenceModel; overwrite::Bool = false)

Run the single-column model (SCM) for a reference model object
using default parameters.

Inputs:
 - m                    :: A `ReferenceModel`.
 - overwrite            :: if true, overwrite existing simulation files.
 - run_single_timestep  :: if true, run only one time step.
 - namelist_args        :: Additional arguments passed to the TurbulenceConvection namelist.
"""
function run_reference_SCM(
    m::ReferenceModel;
    overwrite::Bool = false,
    run_single_timestep = true,
    namelist_args = nothing,
)
    output_dir = scm_dir(m)
    if ~isdir(output_dir) | overwrite
        namelist = get_scm_namelist(m, overwrite = overwrite)
        # Set optional namelist args
        if !isnothing(namelist_args)
            for namelist_arg in namelist_args
                group, name, val = namelist_arg
                namelist[group][name] = val
            end
        end
        default_t_max = namelist["time_stepping"]["t_max"]
        default_adapt_dt = namelist["time_stepping"]["adapt_dt"]
        if run_single_timestep
            # Run only 1 timestep -- since we don't need output data, only simulation config
            namelist["time_stepping"]["adapt_dt"] = false
            namelist["time_stepping"]["t_max"] = namelist["time_stepping"]["dt_min"]
        end
        namelist["meta"]["uuid"] = uuid(m)
        namelist["output"]["output_root"] = dirname(output_dir)
        # if `LES_driven_SCM` case, provide input LES stats file
        if m.case_name == "LES_driven_SCM"
            namelist["meta"]["lesfile"] = get_stats_path(y_dir(m))
        end
        # run TurbulenceConvection.jl
        try
            main(namelist)
        catch
            @warn "Default TurbulenceConvection.jl simulation $(basename(m.y_dir)) failed."
        end
        if run_single_timestep
            # reset t_max to default and overwrite stored namelist file
            namelist["time_stepping"]["t_max"] = default_t_max
            namelist["time_stepping"]["adapt_dt"] = default_adapt_dt
            open(namelist_directory(output_dir, m), "w") do io
                JSON.print(io, namelist, 4)
            end
        end
    end
end


"""
    run_SCM_handler(
        m::ReferenceModel,
        tmpdir::String,
        u::Array{FT, 1},
        u_names::Array{String, 1},
        namelist_args = nothing,
    ) where {FT<:AbstractFloat}

Run a case using a set of parameters `u_names` with values `u`,
and return directory pointing to where data is stored for simulation run.

Inputs:
 - m             :: Reference model
 - tmpdir        :: Temporary directory to store simulation results in
 - u             :: Values of parameters to be used in simulations.
 - u_names       :: SCM names for parameters `u`.
 - namelist_args :: Additional arguments passed to the TurbulenceConvection namelist.
Outputs:
 - output_dirs   :: directory containing output data from the SCM run.
"""
function run_SCM_handler(
    m::ReferenceModel,
    tmpdir::String,
    u::Vector{FT},
    u_names::Vector{String},
    namelist_args = nothing,
) where {FT <: AbstractFloat}
    model_error = false
    # fetch default namelist
    inputdir = scm_dir(m)
    namelist = JSON.parsefile(namelist_directory(inputdir, m))

    u_names, u = create_parameter_vectors(u_names, u)

    # Set optional namelist args
    if !isnothing(namelist_args)
        for namelist_arg in namelist_args
            group, name, val = namelist_arg
            namelist[group][name] = val
        end
    end

    # update parameter values
    for (pName, pVal) in zip(u_names, u)
        namelist["turbulence"]["EDMF_PrognosticTKE"][pName] = pVal
    end

    # set random uuid
    uuid = basename(tmpdir)
    namelist["meta"]["uuid"] = uuid
    # set output dir to `tmpdir`
    namelist["output"]["output_root"] = tmpdir

    # run TurbulenceConvection.jl with modified parameters
    try
        main(namelist)
    catch
        model_error = true
        message = ["TurbulenceConvection.jl simulation $(basename(m.y_dir)) failed with parameters: \n"]
        append!(message, ["$param_name = $param_value \n" for (param_name, param_value) in zip(u_names, u)])
        @warn join(message)
    end
    return data_directory(tmpdir, m.case_name, uuid), model_error
end

"""
    create_parameter_vectors(u_names::Vector{String}, u::Vector{FT}) where {FT <: AbstractFloat}

Given vector of parameter names and corresponding values, combine any vector components
into single parameter vectors for input into SCM.

Inputs:
    u_names :: SCM names for parameters `u`, which may contain vector components.
    u :: Values of parameters to be used in simulations, which may contain vector components.
Outputs:
    u_names_out :: SCM names for parameters `u`.
    u_out :: Values of parameters to be used in simulations.
"""
function create_parameter_vectors(u_names::Vector{String}, u::Vector{FT}) where {FT <: AbstractFloat}

    u_names_out = []
    u_out = []
    u_vec_dict = Dict()

    for i in 1:length(u_names)
        param_name_i = u_names[i]
        if occursin(r"{?}", param_name_i)
            param_name_split = split(param_name_i, "_")
            param_vec_name = join(param_name_split[1:(length(param_name_split) - 1)], "_")
            if param_vec_name in keys(u_vec_dict)
                push!(u_vec_dict[param_vec_name], u[i])
            else
                u_vec_dict[param_vec_name] = [u[i]]
            end
        else
            push!(u_names_out, u_names[i])
            push!(u_out, u[i])

        end
    end
    append!(u_names_out, collect(keys(u_vec_dict)))
    append!(u_out, collect(values(u_vec_dict)))

    return (u_names_out, u_out)
end

"""
    generate_scm_input(
        model_evaluators::Vector{ModelEvaluator{FT}},
        outdir_path::String = pwd(),
    ) where {FT <: AbstractFloat}

Writes to file a set of ModelEvaluator used to initialize SCM
evaluations at different parameter vectors, as well as their
assigned numerical version.
"""
function generate_scm_input(
    model_evaluators::Vector{ModelEvaluator{FT}},
    outdir_path::String = pwd(),
) where {FT <: AbstractFloat}
    # Generate versions conditioned on being unique within the batch.
    used_versions = Vector{Int}()
    for model_evaluator in model_evaluators
        version = rand(11111:99999)
        while version in used_versions
            version = rand(11111:99999)
        end
        jldsave(scm_init_path(outdir_path, version); model_evaluator, version)
        push!(used_versions, version)
    end
    return used_versions
end

"""
    get_gcm_les_uuid(
        cfsite_number::Integer;
        forcing_model::String,
        month::Integer,
        experiment::String,)
Generate unique and self-describing uuid given information about a GCM-driven LES simulation from `Shen et al. 2021`.
"""
function get_gcm_les_uuid(
    cfsite_number::Integer;
    forcing_model::String = "HadGEM2-A",
    month::Integer = 7,
    experiment::String = "amip",
)
    cfsite_number = string(cfsite_number)
    month = string(month, pad = 2)
    return join([cfsite_number, forcing_model, month, experiment], '_')
end

""" Save full EDMF data from every ensemble"""
function save_full_ensemble_data(save_path, sim_dirs_arr, ref_models)
    # get a simulation directory `.../Output.SimName.UUID`, and corresponding parameter name
    for (ens_i, sim_dirs) in enumerate(sim_dirs_arr)  # each ensemble returns a list of simulation directories
        ens_i_path = joinpath(save_path, "ens_$ens_i")
        mkpath(ens_i_path)
        for (ref_model, sim_dir) in zip(ref_models, sim_dirs)
            scm_name = ref_model.case_name
            # Copy simulation data to output directory
            dirname = splitpath(sim_dir)[end]
            @assert dirname[1:7] == "Output."  # sanity check
            # Stats file
            tmp_data_path = joinpath(sim_dir, "stats/Stats.$scm_name.nc")
            save_data_path = joinpath(ens_i_path, "Stats.$scm_name.$ens_i.nc")
            cp(tmp_data_path, save_data_path)
            # namefile
            tmp_namefile_path = namelist_directory(sim_dir, scm_name)
            save_namefile_path = namelist_directory(ens_i_path, scm_name)
            cp(tmp_namefile_path, save_namefile_path)
        end
    end
end

"""
    precondition(
        param::Vector{FT},
        priors,
        ref_models::Vector{ReferenceModel},
        ref_stats::ReferenceStatistics,
        namelist_args = nothing;
        counter::Integer = 0,
        max_counter::Integer = 10,
    ) where {FT <: Real}

Substitute parameter vector `param` by a parameter vector drawn
from the same prior, conditioned on the forward model being stable.

Inputs:
 - param      :: A parameter vector that may possibly result in unstable
    forward model evaluations (in unconstrained space).
 - priors      :: Priors from which the parameters were drawn.
 - ref_models  :: Vector of ReferenceModels to check stability for.
 - ref_stats   :: ReferenceStatistics of the ReferenceModels.
 - namelist_args :: Arguments passed to the TC.jl namelist.
 - counter :: Accumulator tracking number of recursive calls to preconditioner.
 - max_counter :: Maximum number of recursive calls to the preconditioner.
Outputs:
 - new_param  :: A new parameter vector drawn from the prior, conditioned on
  simulations being stable (in unconstrained space).
"""
function precondition(
    param::Vector{FT},
    priors,
    ref_models::Vector{ReferenceModel},
    ref_stats::ReferenceStatistics,
    namelist_args = nothing;
    counter::Integer = 0,
    max_counter::Integer = 10,
) where {FT <: Real}
    param_names = priors.names
    # Wrapper around SCM
    g_(u::Array{Float64, 1}) =
        run_SCM(u, param_names, ref_models, ref_stats, error_check = true, namelist_args = namelist_args)

    param_cons = deepcopy(transform_unconstrained_to_constrained(priors, param))
    _, _, _, model_error = g_(param_cons)
    if model_error && counter < max_counter
        message = ["Unstable parameter vector found: \n"]
        append!(message, ["$param_name = $param \n" for (param_name, param) in zip(param_names, param_cons)])
        @warn join(message)
        @warn "Sampling new parameter vector from prior..."
        return precondition(
            vec(construct_initial_ensemble(priors, 1)),
            priors,
            ref_models,
            ref_stats,
            namelist_args,
            counter = counter + 1,
            max_counter = max_counter,
        )
    elseif model_error
        throw(OverflowError("Number of recursive calls to preconditioner exceeded $(max_counter). Terminating."))
    else
        @info "Preconditioning finished."
        return param
    end
end

"""
    precondition(ME::ModelEvaluator, priors)

Substitute the parameter vector of a ModelEvaluator by another
one drawn from the given `priors`, conditioned on the forward
model being stable.

Inputs:
 - ME          :: A ModelEvaluator.
 - priors      :: Priors from which the parameters were drawn.
Outputs:
 - A preconditioned ModelEvaluator.
"""
function precondition(ME::ModelEvaluator, priors; namelist_args = nothing)
    # Precondition in unconstrained space
    u_orig = transform_constrained_to_unconstrained(priors, ME.param_cons)
    u = precondition(u_orig, priors, ME.ref_models, ME.ref_stats, namelist_args)
    # Transform back to constrained space
    param_cons = transform_unconstrained_to_constrained(priors, u)
    return ModelEvaluator(param_cons, ME.param_names, ME.ref_models, ME.ref_stats)
end

end # module
