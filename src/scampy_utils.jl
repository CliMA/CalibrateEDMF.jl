"""Interaction and postprocessing utils for CliMA/SCAMPy."""

using Base
using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob
using JSON
using Random
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributionStorage


"""
    run_SCAMPy(
        u::Array{FT, 1},
        u_names::Array{String, 1},
        y_names::Union{Array{String, 1}, Array{Array{String,1},1}},
        scampy_dir::String,
        scm_data_root::String,
        scm_names::Array{String, 1},
        ti::Union{Array{FT,1}, Array{Array{FT,1},1}},
        tf::Union{Array{FT,1}, Array{Array{FT,1},1}, Nothing} = nothing;
        norm_var_list = nothing,
        P_pca_list = nothing,
    ) where {FT<:AbstractFloat}

Run call_SCAMPy.sh using a set of parameters u and return
the value of outputs defined in y_names, possibly after
normalization and projection onto lower dimensional space
using PCA.

Inputs:
 - u                :: Values of parameters to be used in simulations.
 - u_names          :: SCAMPy names for parameters `u`.
 - y_names          :: Name of outputs requested for each flow configuration.
 - scampy_dir       :: Path to SCAMPy directory
 - scm_data_root    :: Path to input data for the SCM model.
 - scm_names        :: Names of SCAMPy cases
 - ti               :: Vector of starting times for observation intervals. 
                        If `tf=nothing`, snapshots at `ti` are returned.
 - tf               :: Vector of ending times for observation intervals.
 - norm_var_list    :: Pooled variance vectors. If given, use to normalize output.
 - P_pca_list       :: Vector of projection matrices `P_pca` for each flow configuration.
Outputs:
 - sim_dirs         :: Vector of simulation output directories
 - g_scm            :: Vector of model evaluations concatenated for all flow configurations.
 - g_scm_pca        :: Projection of `g_scm` onto principal subspace spanned by eigenvectors.
"""
function run_SCAMPy(
        u::Array{FT, 1},
        u_names::Array{String, 1},
        y_names::Union{Array{String, 1}, Array{Array{String,1},1}},
        scampy_dir::String,
        scm_data_root::String,
        scm_names::Array{String, 1},
        ti::Union{Array{FT,1}, Array{Array{FT,1},1}},
        tf::Union{Array{FT,1}, Array{Array{FT,1},1}, Nothing} = nothing;
        norm_var_list = nothing,
        P_pca_list = nothing,
    ) where {FT<:AbstractFloat}

    # Check parameter dimensionality
    @assert length(u_names) == length(u)

    # run SCAMPy and get simulation dirs
    sim_dirs = run_SCAMPy_handler(u, u_names, scampy_dir, scm_names, scm_data_root)

    # Check consistent time interval dims
    @assert length(ti) == length(sim_dirs)

    g_scm = zeros(0)
    g_scm_pca = zeros(0)

    if typeof(ti) == Array{FT,1} # 1 interval per simulation
        for (i, sim_dir) in enumerate(sim_dirs)
            ti_ = ti[i]
            tf_ = !isnothing(tf) ? tf[i] : nothing
            y_names_ = typeof(y_names)==Array{Array{String,1},1} ? y_names[i] : y_names

            g_scm_flow = get_profile(sim_dir, y_names_, ti = ti_, tf = tf_)
            if !isnothing(norm_var_list)
                g_scm_flow = normalize_profile(g_scm_flow, length(y_names_), norm_var_list[i])
            end
            append!(g_scm, g_scm_flow)
            if !isnothing(P_pca_list)
                append!(g_scm_pca, P_pca_list[i]' * g_scm_flow)
            end
        end
    elseif typeof(ti) == Array{Array{FT,1},1} # multiple intervals per simulation
        config_num = 1
        for (i, sim_dir) in enumerate(sim_dirs)
            y_names_ = typeof(y_names)==Array{Array{String,1},1} ? y_names[i] : y_names
            for (j, ti_j) in enumerate(ti[i]) # Loop on time intervals per sim
                tf_j = !isnothing(tf) ? tf[i][j] : nothing
                g_scm_flow = get_profile(sim_dir, y_names_, ti = ti_j, tf = tf_j)
                if !isnothing(norm_var_list)
                    g_scm_flow = normalize_profile(g_scm_flow, length(y_names_), norm_var_list[config_num])
                end
                append!(g_scm, g_scm_flow)
                if !isnothing(P_pca_list)
                    append!(g_scm_pca, P_pca_list[config_num]' * g_scm_flow)
                end
                config_num += 1
            end
        end
    end
    # penalize nan-values in output
    for i in eachindex(g_scm)
        g_scm[i] = isnan(g_scm[i]) ? 1.0e5 : g_scm[i]
    end
    if !isnothing(P_pca_list)
        for i in eachindex(g_scm_pca)
            g_scm_pca[i] = isnan(g_scm_pca[i]) ? 1.0e5 : g_scm_pca[i]
        end
        println("LENGTH OF G_SCM_ARR : ", length(g_scm))
        println("LENGTH OF G_SCM_ARR_PCA : ", length(g_scm_pca))
        return sim_dirs, g_scm, g_scm_pca
    else
        return sim_dirs, g_scm
    end
end


"""
    function run_SCAMPy_handler(
        u::Array{FT, 1},  
        u_names::Array{String, 1},
        scampy_dir::String,
        scm_names::String,
        scm_data_root::String,
    ) where {FT<:AbstractFloat}

Run a list of cases using a set of parameters `u_names` with values `u`,
and return a list of directories pointing to where data is stored for 
each simulation run.

Inputs:
 - u :: Values of parameters to be used in simulations.
 - u_names :: SCAMPy names for parameters `u`.
 - scampy_dir :: Path to SCAMPy directory
 - scm_names :: Names of SCAMPy cases to run
 - scm_data_root :: Path to SCAMPy case data (<scm_data_root>/Output.<scm_name>.00000)
Outputs:
 - output_dirs :: list of directories containing output data from the SCAMPy runs.
"""
function run_SCAMPy_handler(
        u::Array{FT, 1},
        u_names::Array{String, 1},
        scampy_dir::String,
        scm_names::Array{String, 1},
        scm_data_root::String,
    ) where {FT<:AbstractFloat}
    # create temporary directory to store SCAMPy data in
    tmpdir = mktempdir(pwd())

    # output directories
    output_dirs = String[]

    for simname in scm_names
        # For each scm case, fetch namelist and paramlist
        inputdir = joinpath(scm_data_root, "Output.$simname.00000")
        namelist = JSON.parsefile(joinpath(inputdir, "$simname.in"))
        paramlist = JSON.parsefile(joinpath(inputdir, "paramlist_$simname.in"))

        # update parameter values
        for (pName, pVal) in zip(u_names, u)
            paramlist["turbulence"]["EDMF_PrognosticTKE"][pName] = pVal
        end
        # write updated paramlist to `tmpdir`
        paramlist_path = joinpath(tmpdir, "paramlist_$simname.in")
        open(paramlist_path, "w") do io
            JSON.print(io, paramlist, 4)
        end

        # generate random uuid
        uuid_end = randstring(5)
        uuid_start = namelist["meta"]["uuid"][1:end-5]
        namelist["meta"]["uuid"] = "$uuid_start$uuid_end"
        # set output dir to `tmpdir`
        namelist["output"]["output_root"] = string(tmpdir, "/")
        # write updated namelist to `tmpdir`
        namelist_path = joinpath(tmpdir, "$simname.in")
        open(namelist_path, "w") do io
            JSON.print(io, namelist, 4)
        end

        # run SCAMPy with modified parameters
        main_path = joinpath(scampy_dir, "main.py")
        Base.run(`python $main_path $namelist_path $paramlist_path`)

        push!(output_dirs, joinpath(tmpdir, "Output.$simname.$uuid_end"))
    end  # end `simnames` loop
    return output_dirs
end


"""
    precondition_ensemble!(params::Array{FT, 2}, priors, 
        param_names::Vector{String}, ::Union{Array{String, 1}, Array{Array{String,1},1}}, 
        ti::Union{FT, Array{FT,1}}, tf::Union{FT, Array{FT,1}};
        lim::FT=1.0e3,) where {IT<:Int, FT}

Substitute all unstable parameters by stable parameters drawn from 
the same prior.
"""
function precondition_ensemble!(params::Array{FT, 2}, priors,
    param_names::Vector{String}, y_names::Union{Array{String, 1}, Array{Array{String,1},1}},
    scampy_dir::String, scm_data_root::String, scm_names::Array{String, 1},
    ti::Union{FT, Array{FT,1}};
    tf::Union{FT, Array{FT,1}, Nothing}=nothing, lim::FT=1.0e4,) where {IT<:Int, FT}

    # Check dimensionality
    @assert length(param_names) == size(params, 1)
    # Wrapper around SCAMPy in original output coordinates
    g_(x::Array{Float64,1}) = run_SCAMPy(
        x, param_names, y_names, scampy_dir, scm_data_root, scm_names, ti, tf,
    )

    params_cons_i = deepcopy(transform_unconstrained_to_constrained(priors, params))    
    params_cons_i = [row[:] for row in eachrow(params_cons_i')]
    N_ens = size(params_cons_i, 1)
    g_ens_arr = pmap(g_, params_cons_i) # [N_ens N_output]
    @assert size(g_ens_arr, 1) == N_ens
    N_out = size(g_ens_arr, 2)
    # If more than 1/4 of outputs are over limit lim, deemed as unstable simulation
    uns_vals_frac = sum(count.(x->x>lim, g_ens_arr), dims=2)./N_out
    unstable_point_inds = findall(x->x>0.25, uns_vals_frac)
    println(string("Unstable parameter indices: ", unstable_point_inds))
    # Recursively eliminate all unstable parameters
    if !isempty(unstable_point_inds)
        println(length(unstable_point_inds), " unstable parameters found:" )
        for j in 1:length(unstable_point_inds)
            println(params[:, unstable_point_inds[j]])
        end
        println("Sampling new parameters from prior...")
        new_params = construct_initial_ensemble(priors, length(unstable_point_inds))
        precondition_ensemble!(new_params, priors, param_names,
            y_names, scampy_dir, scm_data_root, scm_names, ti, tf=tf, lim=lim)
        params[:, unstable_point_inds] = new_params
    end
    println("\nPreconditioning finished.")
    return
end


# Handler for PadeOps data files
function interp_padeops(padeops_data,
                    padeops_z,
                    padeops_t,
                    z_scm,
                    t_scm
                    )
    # Weak verification of limits for independent vars 
    @assert abs(padeops_z[end] - z_scm[end])/padeops_z[end] <= 0.1
    @assert abs(padeops_z[end] - z_scm[end])/z_scm[end] <= 0.1

    # Create interpolating function
    padeops_itp = interpolate( (padeops_t, padeops_z), padeops_data,
                ( Gridded(Linear()), Gridded(Linear()) ) )
    return padeops_itp(t_scm, z_scm)
end


function padeops_m_σ2(padeops_data,
                    padeops_z,
                    padeops_t,
                    z_scm,
                    t_scm,
                    dims_ = 1)
    padeops_snapshot = interp_padeops(padeops_data,padeops_z,padeops_t,z_scm, t_scm)
    # Compute variance along axis dims_
    padeops_var = cov(padeops_data, dims=dims_)
    return padeops_snapshot, padeops_var
end
