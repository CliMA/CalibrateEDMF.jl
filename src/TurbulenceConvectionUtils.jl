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
export generate_scm_input

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
    run_SCM_handler(
        m::ReferenceModel,
        tmpdir::String,
        u::Array{FT, 1},
        u_names::Array{String, 1},
    ) where {FT<:AbstractFloat}

Run a list of cases using a set of parameters `u_names` with values `u`,
and return a list of directories pointing to where data is stored for 
each simulation run.

Inputs:
 - m            :: Reference model
 - tmpdir       :: Directory to store simulation results in
 - u            :: Values of parameters to be used in simulations.
 - u_names      :: SCM names for parameters `u`.
Outputs:
 - output_dirs  :: list of directories containing output data from the SCM runs.
"""
function run_SCM_handler(
    m::ReferenceModel,
    tmpdir::String,
    u::Array{FT, 1},
    u_names::Array{String, 1},
) where {FT <: AbstractFloat}

    # fetch default namelist
    inputdir = m.scm_dir
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
    # write updated namelist to `tmpdir`
    namelist_path = namelist_directory(tmpdir, m)
    open(namelist_path, "w") do io
        JSON.print(io, namelist, 4)
    end

    # run TurbulenceConvection.jl with modified parameters
    main(namelist)

    return data_directory(tmpdir, m.scm_name, uuid)
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


end # module
