module TurbulenceConvectionUtils

using Glob
using JLD2
using JSON
using Random
using ..ReferenceModels
using ..ReferenceStats
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributionStorage
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using TurbulenceConvection
include(joinpath(@__DIR__, "helper_funcs.jl"))

export ModelEvaluator
export run_SCM, run_SCM_handler
export generate_scm_input, get_gcm_les_uuid
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
        RS::ReferenceStatistics,
    ) where FT<:Real

Run the single-column model (SCM) using a set of parameters u 
and return the value of outputs defined in y_names, possibly 
after normalization and projection onto lower dimensional 
space using PCA.

Inputs:
 - u           :: Values of parameters to be used in simulations.
 - u_names     :: SCM names for parameters `u`.
 - RM          :: Vector of `ReferenceModel`s
 - RS          :: reference statistics for simulation
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
) where {FT <: Real}

    g_scm = zeros(0)
    g_scm_pca = zeros(0)
    sim_dirs = String[]

    mkpath(joinpath(pwd(), "tmp"))
    model_error = false
    for (i, m) in enumerate(RM)
        # create temporary directory to store SCM data in
        tmpdir = mktempdir(joinpath(pwd(), "tmp"))

        # run TurbulenceConvection.jl. Get output directory for simulation data
        sim_dir, sim_error = run_SCM_handler(m, tmpdir, u, u_names, namelist_args)
        model_error = model_error || sim_error
        push!(sim_dirs, sim_dir)

        g_scm_flow = get_profile(m, sim_dir, z_scm = get_height(sim_dir))
        # normalize
        g_scm_flow = normalize_profile(g_scm_flow, length(m.y_names), RS.norm_vec[i])
        append!(g_scm, g_scm_flow)

        # perform PCA reduction
        append!(g_scm_pca, RS.pca_vec[i]' * g_scm_flow)
    end

    # penalize nan-values in output
    any(isnan.(g_scm)) && warn("NaN-values in output data")
    g_scm[isnan.(g_scm)] .= 1e5

    g_scm_pca[isnan.(g_scm_pca)] .= 1e5
    @info "Length of g_scm (full): $(length(g_scm))"
    @info "Length of g_scm (pca) : $(length(g_scm_pca))"
    if error_check
        return sim_dirs, g_scm, g_scm_pca, model_error
    else
        return sim_dirs, g_scm, g_scm_pca
    end
end

function run_SCM(ME::ModelEvaluator; error_check::Bool = false, namelist_args = nothing) where {FT <: Real}
    return run_SCM(
        ME.param_cons,
        ME.param_names,
        ME.ref_models,
        ME.ref_stats,
        error_check = error_check,
        namelist_args = namelist_args,
    )
end

"""
    run_SCM(
        RM::Vector{ReferenceModel};
        overwrite::Bool,
    ) where FT<:Real

Run the single-column model (SCM) for each reference model object
using default parameters.

Inputs:
 - RM               :: Vector of `ReferenceModel`s
 - overwrite       :: if true, overwrite existing simulation files
Outputs:
 - Nothing
"""
function run_SCM(RM::Vector{ReferenceModel}; overwrite::Bool = false) where {FT <: Real}

    for ref_model in RM
        output_dir = scm_dir(ref_model)
        if ~isdir(output_dir) | overwrite
            run_SCM_handler(ref_model, dirname(output_dir))
        end
    end
end


"""
    run_SCM_handler(
        m::ReferenceModel,
        tmpdir::String,
        u::Array{FT, 1},
        u_names::Array{String, 1},
    ) where {FT<:AbstractFloat}

Run a case using a set of parameters `u_names` with values `u`,
and return directory pointing to where data is stored for simulation run.

Inputs:
 - m            :: Reference model
 - tmpdir       :: Temporary directory to store simulation results in
 - u            :: Values of parameters to be used in simulations.
 - u_names      :: SCM names for parameters `u`.
Outputs:
 - output_dirs  :: directory containing output data from the SCM run.
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

    # update parameter values
    for (pName, pVal) in zip(u_names, u)
        namelist["turbulence"]["EDMF_PrognosticTKE"][pName] = pVal
    end

    # set random uuid
    uuid = basename(tmpdir)
    namelist["meta"]["uuid"] = uuid
    # set output dir to `tmpdir`
    namelist["output"]["output_root"] = tmpdir

    # Set additional namelist args
    if !isnothing(namelist_args)
        for namelist_arg in namelist_args
            group, name, val = namelist_arg
            namelist[group][name] = val
        end
    end

    # run TurbulenceConvection.jl with modified parameters
    try
        main(namelist)
    catch
        model_error = true
        @warn "TurbulenceConvection.jl simulation failed with parameters:"
        [@warn "$param_name = $param_value" for (param_name, param_value) in zip(u_names, u)]
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
    run_SCM_handler(
        m::ReferenceModel,
        output_dir::String;
    ) where {FT<:AbstractFloat}

Run a case with default SCM parameters and return data
directory pointing to where data is stored for simulation run.

Inputs:
 - m            :: Reference model
 - output_dir   :: Directory to store simulation results in
Outputs:
 - output_dirs  :: directory containing output data from the SCM run.
"""
function run_SCM_handler(m::ReferenceModel, output_dir::String) where {FT <: AbstractFloat}

    namelist = NameList.default_namelist(m.case_name)
    # calling NameList.default_namelist writes namelist to pwd
    rm("namelist_" * namelist["meta"]["casename"] * ".in")
    namelist["meta"]["uuid"] = uuid(m)
    # set output dir to `output_dir`
    namelist["output"]["output_root"] = output_dir
    # if `LES_driven_SCM` case, provide input LES stats file
    if m.case_name == "LES_driven_SCM"
        namelist["meta"]["lesfile"] = get_stats_path(y_dir(m))
    end
    # run TurbulenceConvection.jl
    try
        main(namelist)
    catch
        @warn "Default TurbulenceConvection.jl simulation failed. Verify default setup."
    end
    return data_directory(output_dir, m.case_name, namelist["meta"]["uuid"])
end

"""
    generate_scm_input(model_evaluator::ModelEvaluator, outdir_path::String = pwd())

Generates all data necessary to initialize a SCM evaluation
at the given parameter vector `u`.
"""
function generate_scm_input(model_evaluator::ModelEvaluator, outdir_path::String = pwd())
    # Generate version
    version = rand(11111:99999)
    jldsave(scm_init_path(outdir_path, version); model_evaluator, version)
    return version
end
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
    ) where {FT <: Real}

Substitute parameter vector `param` by a parameter vector drawn
from the same prior, conditioned on the forward model being stable.

Inputs:
 - param      :: A parameter vector that may possibly result in unstable
    forward model evaluations (in unconstrained space).
 - priors      :: Priors from which the parameters were drawn.
 - ref_models  :: Vector of ReferenceModels to check stability for.
 - ref_stats   :: ReferenceStatistics of the ReferenceModels.
Outputs:
 - new_param  :: A new parameter vector drawn from the prior, conditioned on
  simulations being stable (in unconstrained space).
"""
function precondition(
    param::Vector{FT},
    priors,
    ref_models::Vector{ReferenceModel},
    ref_stats::ReferenceStatistics,
    namelist_args = nothing,
) where {FT <: Real}
    param_names = priors.names
    # Wrapper around SCM
    g_(u::Array{Float64, 1}) =
        run_SCM(u, param_names, ref_models, ref_stats, error_check = true, namelist_args = namelist_args)

    param_cons = deepcopy(transform_unconstrained_to_constrained(priors, param))
    _, _, _, model_error = g_(param_cons)
    if model_error
        @warn "Unstable parameter vector found:"
        [@warn "$param_name = $param" for (param_name, param) in zip(param_names, param)]
        @warn "Sampling new parameter vector from prior..."
        new_param =
            precondition(vec(construct_initial_ensemble(priors, 1)), priors, ref_models, ref_stats, namelist_args)
    else
        new_param = param
        @info "Preconditioning finished."
    end
    return new_param
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
