using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(dirname(dirname(@__DIR__)))

@everywhere begin
    using CalibrateEDMF
    using CalibrateEDMF.HelperFuncs
    using CalibrateEDMF.ReferenceStats
    using CalibrateEDMF.LESUtils
    using CalibrateEDMF.ReferenceModels
    import CalibrateEDMF.ModelTypes: LES
    import CalibrateEDMF.Pipeline: get_ref_model_kwargs, get_ref_stats_kwargs
    import LinearAlgebra: dot
    import Statistics: mean
    import JSON
end

using Combinatorics
import NCDatasets
const NC = NCDatasets
using ArgParse
using Glob

@everywhere begin
    "Get index of a case, defined local to each case name"
    get_case_ind(cases, case_i) = length(cases[1:case_i][(cases .== cases[case_i])[1:case_i]])

    """
        compute_loss(t) = compute_loss(t...)
        compute_loss(value1::Number, value2::Number, case_ind::I, ens_ind::I, nt::NamedTuple) where {I <: Integer}

    Compute loss function from forward model output data.

    Arguments:
    value1      :: a parameter value (in config > grid_search > parameters > param_name_1)
    value2      :: a parameter value (in config > grid_search > parameters > param_name_2)
    case_ind    :: index of a case name (in config > reference/validation > case_name)
    ens_ind     :: index of the ensemble member
    nt          :: Named tuple that contains information that is constant across forward model data.
        Has the entries `sim_dir`, `group_name`, `config`, `sim_type`, `ref_stats`, `ref_models`, denoting
        the root path to forward model data, the parameter pair name, the config file, the simulation type
        (reference or validation), the reference statistics, and the reference models, respectively.

    """
    compute_loss(t) = compute_loss(t...)
    function compute_loss(value1::Number, value2::Number, case_ind::I, ens_ind::I, nt::NamedTuple) where {I <: Integer}
        # fetch from config
        get_from_config(x) = nt.config[nt.sim_type][x][case_ind]
        (case_name, loss_names, y_dir) = get_from_config.(["case_name", "y_names", "y_dir"])
        y_ref_type = nt.config[nt.sim_type]["y_reference_type"]
        # Get the case index for this case (if there are duplicate cases)
        cases = nt.config[nt.sim_type]["case_name"]
        case_j = get_case_ind(cases, case_ind)

        # path to .nc simulation data
        param_path = joinpath(nt.sim_dir, nt.group_name, "$(value1)_$(value2)")
        @warn("path $param_path")
        #output_dir = joinpath(param_path, "$case_name.$case_j/Output.$case_name.$ens_ind")
        #scm_file = joinpath(output_dir, "stats/Stats.$case_name.nc")
        #if !isfile(scm_file)
        #    @warn("No NetCDF file found on path $scm_file")
        #    return NaN  # if case data does not exist, return NaN
        #end
        # compute loss
        #y_loss_names = if (y_ref_type isa LES)
        #    get_les_names(loss_names, get_stats_path(y_dir))
        #else
        #    loss_names
        #end
        
        #RS = nt.ref_stats
        #m = nt.ref_models[case_ind]

        # Check that the simulation interval fully covers the averaging interval (i.e. simulation completed)
        #t = nc_fetch(scm_file, "t")
        #ti, tf = get_t_start(m), get_t_end(m)
        #dt = (length(t) > 1) * mean(diff(t))
        #if (t[end] < ti) || (t[end] < tf - dt)
            # check is simulation completed to t_max in the namelist
        #    namelist_path = joinpath(output_dir, "namelist_$case_name.in")
        #    namelist = open(namelist_path, "r") do io
        #        JSON.parse(io; dicttype = Dict, inttype = Int64)
        #    end
        #    t_max = namelist["time_stepping"]["t_max"]
        #    if t[end] < t_max
        #        @warn "Simulation did not finish during grid search deleting output directory: \n $output_dir"
        #        rm(output_dir, force=true, recursive=true)
        #    end
#
 #           @warn string(
 #               "The requested averaging interval: ($ti s, $tf s) is not in the simulation interval ($(t[1]) s, $(t[end]) s), ",
 #               "which means the simulation stopped before reaching the requested t_end. Returning NaN. ",
  #              "\n Simulation file: $scm_file"
   #         )
    #        return NaN
      #  end

        # case PCA and covariance matrix
       # pca_vec = RS.pca_vec[case_ind]'
       # pca_case_inds = pca_inds(RS, case_ind)
       # Γ = RS.Γ[pca_case_inds, pca_case_inds]

        # Reference data
       # y_full_case = get_profile(m, get_stats_path(y_dir), y_loss_names, z_scm = get_z_obs(m))
        #y_norm = normalize_profile(y_full_case, num_vars(m), RS.norm_vec[case_ind])

        # SCM data
        #g_full_case_i = get_profile(m, scm_file, z_scm = get_z_obs(m))
        #g_norm = normalize_profile(g_full_case_i, num_vars(m), RS.norm_vec[case_ind])

        # PCA
        #yg_diff_pca = pca_vec * (y_norm - g_norm)
        # loss
        #sim_loss = dot(yg_diff_pca, Γ \ yg_diff_pca)
        return 0 #sim_loss
    end
end  # end @everywhere begin

"""
    compute_loss_map(config::Dict, sim_dir::AbstractString)

Compute loss map for a set of simulations and store the output in a netcdf4 file

Parameters:
config  :: Dictionary of model calibration and grid search configuration options
sim_dir :: Directory of grid search simulation output
"""
function compute_loss_map(config::Dict, sim_dir::AbstractString)

    # construct reference models and reference statistics
    sim_type = get_entry(config["grid_search"], "sim_type", "reference")
    @assert sim_type in ("reference", "validation")
    ref_config = config[sim_type]  # or validation
    kwargs_ref_model = get_ref_model_kwargs(ref_config)
    reg_config = config["regularization"]
    kwargs_ref_stats = get_ref_stats_kwargs(ref_config, reg_config)

    n_ens = get_entry(config["grid_search"], "ensemble_size", nothing)
    cases = get_entry(config[sim_type], "case_name", nothing)
    case_names_unique = ["$case.$(get_case_ind(cases, i))" for (i, case) in enumerate(cases)]
    n_cases = length(cases)

    parameters = get_entry(config["grid_search"], "parameters", nothing)
    param_names = collect(keys(parameters))
    NC.Dataset(joinpath(sim_dir, "loss_hypercube.nc"), "c") do ds
        for (param_name_1, param_name_2) in combinations(param_names, 2)
            group_name = "$param_name_1.$param_name_2"

            # iterate all folders, e.g. "0.1_0.2", ...
            # compute loss, store in matrix:
            # (p1, p2, case, ens_i)
            value1 = parameters[param_name_1]
            value2 = parameters[param_name_2]
            n_value1 = length(value1)
            n_value2 = length(value2)

            loss_2D_sec = zeros((n_value1, n_value2, n_cases, n_ens))
            
            # by default, we use as `scm_dirs` some fixed directories, that is assumed to exist ..
            group_path = joinpath(sim_dir, group_name)
            if !isdir(group_path)
                @warn "No directory found for parameter pair ($param_name_1, $param_name_2). Skipping ..."
                continue
            end
            param_path = joinpath(group_path, "$(value1[1])_$(value2[1])")
            scm_dirs = joinpath.(param_path, case_names_unique, "Output." .* cases .* ".1")
            not_scm_dirs = @. ~isdir(scm_dirs)
            if any(not_scm_dirs)  # .. if any of the scm case directories do not exist, do a glob search for other candidate paths ..
                alt_candidate_scm_paths = glob.("*/" .* case_names_unique[not_scm_dirs] .* "*/*/stats/*.nc", group_path)
                if any(isempty.(alt_candidate_scm_paths))  # .. if no alternative paths are found for any cases ..
                    @warn "For the parameter pair ($param_name_1, $param_name_2), no forward model data is found for the cases: $(join(case_names_unique[isempty.(alt_candidate_scm_paths)], ", "))"
                    continue  # .. we ignore this parameter pair.
                end
            end
            kwargs_ref_model[:scm_dir] = scm_dirs
            # Construct `ReferenceModel`s and `ReferenceStatistics`
            ref_models = construct_reference_models(kwargs_ref_model)
            ref_stats = ReferenceStatistics(ref_models; kwargs_ref_stats...)

            # configurations for `compute_loss`
            nt = (; sim_dir, group_name, config, sim_type, ref_stats, ref_models)
            loss_configs = vec(collect(Iterators.product(value1, value2, 1:n_cases, 1:n_ens, [nt])))

            # compute loss
            sim_loss = pmap(compute_loss, loss_configs)
            loss_2D_sec = reshape(sim_loss, size(loss_2D_sec))

            # save output
            NC.defGroup(ds, group_name, attrib = [])
            ensemble_member = 1:n_ens
            group_root = ds.group[group_name]  # group is e.g. 'sorting_power.entrainment_factor'
            # Define dimensions: param_name_1, param_name_2, case, ensemble_member
            NC.defDim(group_root, "param_name_1", n_value1)
            NC.defDim(group_root, "param_name_2", n_value2)
            NC.defDim(group_root, "case", n_cases)
            NC.defDim(group_root, "ensemble_member", n_ens)
            # Define variables: 
            ncvar = NC.defVar(group_root, "case", case_names_unique, ("case",))
            ncvar[:] = case_names_unique  # "Bomex", "DYCOMS"
            ncvar = NC.defVar(group_root, "ensemble_member", ensemble_member, ("ensemble_member",))
            ncvar[:] = ensemble_member  # 1, 2, ..., n_ens
            # param_name_1 and param_name_2 values
            ncvar = NC.defVar(group_root, param_name_1, value1, ("param_name_1",))
            ncvar[:] = value1
            ncvar = NC.defVar(group_root, param_name_2, value2, ("param_name_2",))
            ncvar[:] = value2
            ncvar = NC.defVar(
                group_root,
                "loss_data",
                loss_2D_sec,
                ("param_name_1", "param_name_2", "case", "ensemble_member"),
            )
            ncvar[:, :, :, :] = loss_2D_sec
        end  # end for param combinations loop
    end  # NC.Dataset do-block
end


function parse_commandline_lm()

    s = ArgParseSettings(; description = "Run data path input")

    @add_arg_table s begin
        "--sim_dir"
        help = "Grid search simulations directory"
        arg_type = String
    end

    return ArgParse.parse_args(s)  # parse_args(ARGS, s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline_lm()
    sim_dir = args["sim_dir"]
    include(joinpath(sim_dir, "config.jl"))
    config = get_config()
    compute_loss_map(config, sim_dir)
end
