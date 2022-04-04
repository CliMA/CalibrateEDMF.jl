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

function compute_loss_map(config, sims_path, sim_type)
    @assert sim_type in ("reference", "validation")
    n_ens = get_entry(config["grid_search"], "ensemble_size", nothing)
    case_names = get_entry(config[sim_type], "case_name", nothing)
    n_cases = length(case_names)

    # param_paths = filter(isdir, readdir(sims_path, join=true))
    parameters = get_entry(config["grid_search"], "parameters", nothing)
    param_names = collect(keys(parameters))
    NC.Dataset(joinpath(sims_path, "loss_hypercube.nc"), "c") do ds
        for (param_name_1, param_name_2) in combinations(param_names, 2)
            group_name = "$param_name_1.$param_name_2"
            NC.defGroup(ds, group_name, attrib = [])

            # iterate all folders, e.g. "0.1_0.2", ...
            # compute loss, store in matrix:
            # (case, ens_i, p1, p2)
            param_val_1 = parameters[param_name_1]
            param_val_2 = parameters[param_name_2]
            n_param_val_1 = length(param_val_1)
            n_param_val_2 = length(param_val_2)

            loss_2D_sec = zeros((n_param_val_1, n_param_val_2, n_cases, n_ens))

            nt = (; param_val_1, param_val_2, sims_path, group_name, config, sim_type)

            loss_configs = vec(collect(Iterators.product(param_val_1, param_val_2, 1:n_cases, 1:n_ens, [nt])))

            @time begin
                sim_loss = pmap(compute_loss, loss_configs)
                loss_2D_sec = reshape(sim_loss, size(loss_2D_sec))
            end
            ensemble_member = LinRange(1, n_ens, n_ens)
            group_root = ds.group[group_name]  # group is e.g. 'sorting_power.entrainment_factor'
            # Define dimensions: param_name_1, param_name_2, case, ensemble_member
            NC.defDim(group_root, "param_name_1", n_param_val_1)
            NC.defDim(group_root, "param_name_2", n_param_val_2)
            NC.defDim(group_root, "case", n_cases)
            NC.defDim(group_root, "ensemble_member", n_ens)
            # Define variables: 
            ncvar = NC.defVar(group_root, "case", case_names, ("case",))
            ncvar[:] = case_names  # "Bomex", "DYCOMS"
            ncvar = NC.defVar(group_root, "ensemble_member", ensemble_member, ("ensemble_member",))
            ncvar[:] = ensemble_member  # 1, 2, ..., n_ens
            # param_name_1 and param_name_2 values
            ncvar = NC.defVar(group_root, param_name_1, param_val_1, ("param_name_1",))
            ncvar[:] = param_val_1
            ncvar = NC.defVar(group_root, param_name_2, param_val_2, ("param_name_2",))
            ncvar[:] = param_val_2
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

@everywhere begin
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
        Has the entries `param_val_1`, `param_val_2`, `sims_path`, `group_name`, `config`, `sim_type`, denoting
        the values for param_name_1, the values for param_name_2, the root path to forward model data, the 
        parameter pair name, the config file, and the simulation type (reference or validation),
        respectively.
    """
    compute_loss(t) = compute_loss(t...)
    function compute_loss(
        param_val_1::Number,
        param_val_2::Number,
        case_ind::I,
        ens_ind::I,
        nt::NamedTuple,
    ) where {I <: Integer}
        # fetch from config
        get_from_config(x) = nt.config[nt.sim_type][x][case_ind]
        (case_name, loss_names, t_start, t_end, y_dir) =
            get_from_config.(["case_name", "y_names", "t_start", "t_end", "y_dir"])
        y_ref_type = nt.config[nt.sim_type]["y_reference_type"]

        # path to .nc simulation data
        param_path = joinpath(nt.sims_path, nt.group_name, "$(param_val_1)_$(param_val_2)")
        nc_file = joinpath(param_path, "$case_name/Output.$case_name.$ens_ind/stats/Stats.$case_name.nc")
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


### Run loss map script
s = ArgParseSettings()
@add_arg_table s begin
    "--sims_path"
    help = "Path to forward simulation output"
    arg_type = String
    "--sim_type"
    help = "Type of simulations to consider (`reference` or `validation`)"
    arg_type = String
    default = "reference"
end
parsed_args = parse_args(ARGS, s)
sims_path = parsed_args["sims_path"]
sim_type = parsed_args["sim_type"]
include(joinpath(sims_path, "config.jl"))
config = get_config()

compute_loss_map(config, sims_path, sim_type)
