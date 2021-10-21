module TurbulenceConvectionUtils

using Glob
using JLD2
using JSON
using Random
using ..ReferenceModels
using ..ReferenceStats
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributionStorage
include(joinpath(@__DIR__, "helper_funcs.jl"))

export run_SCM, run_SCM_handler
export generate_scm_input, get_gcm_les_uuid

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
 - u                :: Values of parameters to be used in simulations.
 - u_names          :: SCM names for parameters `u`.
 - RM               :: Vector of `ReferenceModel`s
 - RS               :: reference statistics for simulation
Outputs:
 - sim_dirs         :: Vector of simulation output directories
 - g_scm            :: Vector of model evaluations concatenated for all flow configurations.
 - g_scm_pca        :: Projection of `g_scm` onto principal subspace spanned by eigenvectors.
"""
function run_SCM(
    u::Vector{FT},
    u_names::Vector{String},
    RM::Vector{ReferenceModel},
    RS::ReferenceStatistics,
) where {FT <: Real}

    g_scm = zeros(0)
    g_scm_pca = zeros(0)
    sim_dirs = String[]

    mkpath(joinpath(pwd(), "tmp"))
    for (i, m) in enumerate(RM)
        # create temporary directory to store SCM data in
        tmpdir = mktempdir(joinpath(pwd(), "tmp"))

        # run TurbulenceConvection.jl. Get output directory for simulation data
        sim_dir = run_SCM_handler(m, tmpdir, u, u_names)
        push!(sim_dirs, sim_dir)

        g_scm_flow = get_profile(m, sim_dir)
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
    println("LENGTH OF G_SCM_ARR : ", length(g_scm))
    println("LENGTH OF G_SCM_ARR_PCA : ", length(g_scm_pca))
    return sim_dirs, g_scm, g_scm_pca
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
    u::Array{FT, 1},
    u_names::Array{String, 1},
) where {FT <: AbstractFloat}

    # fetch default namelist
    inputdir = scm_dir(m)
    namelist = JSON.parsefile(namelist_directory(inputdir, m))

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
    main(namelist)

    return data_directory(tmpdir, m.case_name, uuid)
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
 - output_dir       :: Directory to store simulation results in
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
        namelist["meta"]["lesfile"] = get_stats_path(les_dir(m))
    end
    # run TurbulenceConvection.jl
    main(namelist)

    return data_directory(output_dir, m.case_name, namelist["meta"]["uuid"])
end

"""
    generate_scm_input(u::Vector{Float64},
        u_names::Vector{String},
        RM::Vector{ReferenceModel},
        RS::ReferenceStatistics,)

Generates all data necessary to initialize a SCM evaluation
at the given parameter vector `u`.
"""
function generate_scm_input(
    u::Vector{Float64},
    u_names::Vector{String},
    RM::Vector{ReferenceModel},
    RS::ReferenceStatistics,
    outdir_path::String = pwd(),
)
    # Generate version
    version = rand(11111:99999)
    ref_models = map(x -> serialize_struct(x), RM)
    ref_stats = serialize_struct(RS)
    jldsave(scm_init_path(outdir_path, version); u, u_names, ref_models, ref_stats, version)
    return version
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


end # module
