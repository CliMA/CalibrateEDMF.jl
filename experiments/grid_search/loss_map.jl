using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(dirname(dirname(@__DIR__)))

@everywhere begin
    using CalibrateEDMF
    using CalibrateEDMF.HelperFuncs
    using CalibrateEDMF.ReferenceStats
    using CalibrateEDMF.LESUtils
    using Combinatorics
    import NCDatasets
    const NC = NCDatasets
    config_path = "config.jl"
    include(config_path)

    function compute_loss_map(
        # config_path,
        sim_type = "reference",
        sims_path = "output/20220328_bNQ/",
    )

        config = get_config()

        n_ens = get_entry(config["grid_search"], "ensemble_size", nothing)
        case_names = get_entry(config[sim_type], "case_name", nothing)
        loss_names = get_entry(config[sim_type], "y_names", nothing)
        t_start = get_entry(config[sim_type], "t_start", nothing)
        t_end = get_entry(config[sim_type], "t_end", nothing)
        y_dir = get_entry(config[sim_type], "y_dir", nothing)
        y_ref_type = get_entry(config[sim_type], "y_reference_type", nothing)

        # getparam combinations
        # param_paths = filter(isdir, readdir(sims_path, join=true))
        parameters = get_entry(config["grid_search"], "parameters", nothing)
        param_names = collect(keys(parameters))
        NC.Dataset(joinpath(sims_path,"loss_hypercube.nc"),"c") do ds
            @time begin
                for (param1, param2) in combinations(param_names, 2)
                    group_name = "$param1.$param2"
                    NC.defGroup(ds, group_name, attrib = [])
                    A = parameters[param1]
                    B = parameters[param2]
                    param_values = vec(collect(Iterators.product(A, B)))

                    # iterate all folders, e.g. "0.1_0.2", ...
                    # compute loss, store in matrix: 
                    # (case, ens_i, p1, p2)
                    pval1 = first.(param_values)
                    pval2 = last.(param_values)

                    n_pval1 = length(pval1)
                    n_pval2 = length(pval2)

                    n_cases = length(case_names)

                    loss = zeros((n_pval1, n_pval2, length(case_names), n_ens))

                    nt = (; pval1, pval2, sims_path, group_name, config, sim_type)

                    loss_configs = vec(collect( Iterators.product(1:n_pval1, 1:n_pval2, 1:n_cases, 1:n_ens, [nt]) ))

                    @time begin
                        sim_loss = pmap(compute_loss, loss_configs)
                        loss = reshape(sim_loss, size(loss))
                    end

                    @time begin
                        # yair - need to create a group for each loss map and store the valus and coordinates
                        ensemble_member = LinRange(1, n_ens, n_ens)
                        group_root = ds.group[group_name]  # group is 'sorting_power.entrainment_factor'
                        # Define dimensions: param1, param2, case, ensemble_member
                        NC.defDim(group_root, "param1", n_pval1)
                        NC.defDim(group_root, "param2", n_pval2)
                        NC.defDim(group_root, "case", length(case_names))
                        NC.defDim(group_root, "ensemble_member", n_ens)
                        # Define variables: 
                        ncvar = NC.defVar(group_root, "case", case_names, ("case",))
                        ncvar[:] = case_names  # "Bomex", "DYCOMS"
                        ncvar = NC.defVar(group_root, "ensemble_member", ensemble_member, ("ensemble_member",))
                        ncvar[:] = ensemble_member  # 1, 2, ..., n_ens
                        # param1 and param2 values
                        ncvar = NC.defVar(group_root, param1, pval1, ("param1",))
                        ncvar[:] = pval1
                        ncvar = NC.defVar(group_root, param2,pval2,("param2",))
                        ncvar[:] = pval2
                        ncvar = NC.defVar(group_root, "loss_data", loss, ("param1", "param2", "case", "ensemble_member"))
                        ncvar[:,:,:,:] = loss
                    end
                end
            end  # time
        end  # do-block
    end

    compute_loss(t) = compute_loss(t...)
    function compute_loss(p1_i, p2_j, case_k, ens_l, nt)
        sim_type = nt.sim_type

        n_ens = get_entry(nt.config["grid_search"], "ensemble_size", nothing)
        case_names = get_entry(nt.config[sim_type], "case_name", nothing)
        loss_names = get_entry(nt.config[sim_type], "y_names", nothing)
        t_start = get_entry(nt.config[sim_type], "t_start", nothing)
        t_end = get_entry(nt.config[sim_type], "t_end", nothing)
        y_dir = get_entry(nt.config[sim_type], "y_dir", nothing)
        y_ref_type = get_entry(nt.config[sim_type], "y_reference_type", nothing)

        p1 = nt.pval1[p1_i]
        p2 = nt.pval2[p2_j]
        param_path = joinpath(nt.sims_path, nt.group_name, "$(p1)_$(p2)")
        case_name = case_names[case_k]
        sim_path = joinpath(param_path, "$case_name/Output.$case_name.$ens_l")
        nc_file = joinpath(sim_path, "stats/Stats.$case_name.nc")
        # compute mse
        z_scm = get_height(nc_file)
        filename = y_dir[case_k]
        y_loss_names = if (y_ref_type == LES)
            get_les_names(loss_names[case_k], filename)
        else
            loss_names[case_k]
        end
        y_full_case = get_profile(filename, y_loss_names, ti = t_start[case_k], tf = t_end[case_k], z_scm = z_scm)
        g_full_case_i = get_profile(
            nc_file, loss_names[case_k], ti = t_start[case_k], tf = t_end[case_k], z_scm = z_scm
        )
        sim_loss = compute_mse([g_full_case_i], y_full_case)[1]
        return sim_loss
    end
end

compute_loss_map()
