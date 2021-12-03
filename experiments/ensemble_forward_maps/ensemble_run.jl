# Import modules to all processes
@everywhere using Pkg
@everywhere Pkg.activate("../..")
@everywhere using Distributions
@everywhere using StatsBase
@everywhere using LinearAlgebra
# Import EKP modules
@everywhere using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
@everywhere using EnsembleKalmanProcesses.Observations
@everywhere using EnsembleKalmanProcesses.ParameterDistributionStorage
@everywhere include(joinpath(@__DIR__, "../../src/helper_funcs.jl"))
using JLD2


function run_ensemble(param_names, param_values, N_ens)

    scm_root = "/groups/esm/hervik/calibration"
    ref_model = ReferenceModel("Bomex",
        scm_parent_dir = scm_root,
        scm_suffix = "stoch_sde_exp",
        # scm_suffix = "lognormal_med",
        t_start = 4.0 * 3600,  # 4hrs
        t_end = 6.0 * 3600,    # 6hrs
        y_names = [""], les_dir = "",
    )
    case_name = ref_model.case_name

    # Define caller function
    @everywhere g_(tmpdir::String) = run_SCM_handler(
        $ref_model, tmpdir, $param_values, $param_names,
    )

    # Generate output folder
    path_suffix = "sde_exp_transform"
    outdir_root = joinpath(pwd(), "output")
    n_params = length(param_names)
    outdir_path = joinpath(outdir_root, "results_ens_p$(n_params)_e$(N_ens)_$(case_name)_$path_suffix")
    sim_folder = "noise$(param_values[1])"
    outdir_path = joinpath(outdir_path, "scm_data", sim_folder)
    mkpath(outdir_path)

    # Run ensemble forward maps
    tmpdirs = [mktempdir(outdir_root) for _ in 1:N_ens]
    @everywhere tmpdirs = $tmpdirs
    raw_output_dirs = pmap(g_, tmpdirs,
        on_error=x->nothing,
    )
    raw_output_dirs = filter(x -> !isnothing(x), raw_output_dirs)
    println("$(length(raw_output_dirs)) / $N_ens ensemble members completed successfully")

    # Save output
    # namefile
    tmp_namefile_path = namelist_directory(raw_output_dirs[1], case_name)
    save_namefile_path = namelist_directory(outdir_path, case_name)
    cp(tmp_namefile_path, save_namefile_path)
    # Stats file
    for (ens_i, raw_output_dir) in enumerate(raw_output_dirs)
        tmp_data_path = joinpath(raw_output_dir, "stats/Stats.$case_name.nc")
        save_data_path = joinpath(outdir_path, "Stats.$case_name.$ens_i.nc")
        cp(tmp_data_path, save_data_path)
    end
end

""" Save full EDMF data from every ensemble"""
function save_full_ensemble_data(save_path, sim_dirs, scm_name)
    # get a simulation directory `.../Output.SimName.UUID`, and corresponding parameter name
    for (ens_i, sim_dirs) in enumerate(sim_dirs_arr)  # each ensemble returns a list of simulation directories
        ens_i_path = joinpath(save_path, "ens_$ens_i")
        mkpath(ens_i_path)
        for (ens_i, sim_dir) in enumerate(sim_dirs)
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