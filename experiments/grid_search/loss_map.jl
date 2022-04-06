using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(dirname(dirname(@__DIR__)))

@everywhere begin
    using CalibrateEDMF
    using CalibrateEDMF.HelperFuncs
    using CalibrateEDMF.ReferenceStats
    using CalibrateEDMF.LESUtils
    import CalibrateEDMF.ModelTypes: LES
    using Combinatorics
    import NCDatasets
    const NC = NCDatasets
    using ArgParse
end

@everywhere begin
    "Get index of a case, defined local to each case name"
    get_case_ind(cases, case_i) = length(cases[1:case_i][(cases .== cases[case_i])[1:case_i]])

    """
        compute_loss(t) = compute_loss(t...)
        compute_loss(param_name_1_ind, param_name_2_ind, case_ind, ens_ind, nt)

    Compute loss function from forward model output data.

    Arguments:
    param_name_1_ind  :: index of a parameter value (in config > grid_search > parameters > param_name_1)
    param_name_2_ind  :: index of a parameter value (in config > grid_search > parameters > param_name_2)
    case_ind    :: index of a case name (in config > reference/validation > case_name)
    ens_ind     :: index of the ensemble member
    nt          :: Named tuple that contains information that is constant across forward model data.
        Has the entries `sim_dir`, `group_name`, `config`, `sim_type`, denoting the root path to 
        forward model data, the parameter pair name, the config file, and the simulation type 
        (reference or validation), respectively.
    """
    compute_loss(t) = compute_loss(t...)
    function compute_loss(value1::Number, value2::Number, case_ind::I, ens_ind::I, nt::NamedTuple) where {I <: Integer}
        # fetch from config
        get_from_config(x) = nt.config[nt.sim_type][x][case_ind]
        (case_name, loss_names, t_start, t_end, y_dir) =
            get_from_config.(["case_name", "y_names", "t_start", "t_end", "y_dir"])
        y_ref_type = nt.config[nt.sim_type]["y_reference_type"]
        # Get the case index for this case (if there are duplicate cases)
        cases = nt.config[nt.sim_type]["case_name"]
        case_j = get_case_ind(cases, case_ind)

        # path to .nc simulation data
        param_path = joinpath(nt.sim_dir, nt.group_name, "$(value1)_$(value2)")
        nc_file = joinpath(param_path, "$case_name.$case_j/Output.$case_name.$ens_ind/stats/Stats.$case_name.nc")
        # compute mse
        z_scm = get_height(nc_file)
        y_loss_names = if (y_ref_type isa LES)
            get_les_names(loss_names, y_dir)
        else
            loss_names
        end
        y_full_case = get_profile(y_dir, y_loss_names, ti = t_start, tf = t_end, z_scm = z_scm)
        g_full_case_i = get_profile(nc_file, loss_names, ti = t_start, tf = t_end, z_scm = z_scm)
        sim_loss = compute_mse([g_full_case_i], y_full_case)[1]
        return sim_loss
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
    sim_type = get_entry(config["grid_search"], "sim_type", "reference")
    @assert sim_type in ("reference", "validation")
    n_ens = get_entry(config["grid_search"], "ensemble_size", nothing)
    cases = get_entry(config[sim_type], "case_name", nothing)
    case_names_unique = ["$case.$(get_case_ind(cases, i))" for (i, case) in enumerate(cases)]
    n_cases = length(cases)

    parameters = get_entry(config["grid_search"], "parameters", nothing)
    param_names = collect(keys(parameters))
    NC.Dataset(joinpath(sim_dir, "loss_hypercube.nc"), "c") do ds
        for (param_name_1, param_name_2) in combinations(param_names, 2)
            group_name = "$param_name_1.$param_name_2"
            NC.defGroup(ds, group_name, attrib = [])

            # iterate all folders, e.g. "0.1_0.2", ...
            # compute loss, store in matrix:
            # (p1, p2, case, ens_i)
            value1 = parameters[param_name_1]
            value2 = parameters[param_name_2]
            n_value1 = length(value1)
            n_value2 = length(value2)

            loss_2D_sec = zeros((n_value1, n_value2, n_cases, n_ens))

            nt = (; sim_dir, group_name, config, sim_type)

            loss_configs = vec(collect(Iterators.product(value1, value2, 1:n_cases, 1:n_ens, [nt])))

            # compute loss
            sim_loss = pmap(compute_loss, loss_configs, on_error = e -> NaN)
            loss_2D_sec = reshape(sim_loss, size(loss_2D_sec))

            # save output
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
        end
    end  # do-block
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
