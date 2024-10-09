using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl")
using CalibrateEDMF
CEDMF_dir = "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl"


# experiments list
experiments = (
    "SOCRATES_Base",
    "SOCRATES_exponential_T_scaling_ice",
    "SOCRATES_exponential_T_scaling_ice_raw",
    "SOCRATES_powerlaw_T_scaling_ice",
    "SOCRATES_geometric_liq__geometric_ice",
    "SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice",
    "SOCRATES_geometric_liq__powerlaw_T_scaling_ice",
    "SOCRATES_neural_network",
    "SOCRATES_linear_combination",
    "SOCRATES_linear_combination_with_w",
)

setups = ("pow_icenuc_autoconv_eq", "tau_autoconv_noneq")



CEDMF_dir = "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl"
CEDMF_data_dir = "/home/jbenjami/Data/Research_Schneider/CliMa/CalibrateEDMF.jl/"

SOCRATES_dir = joinpath(CEDMF_dir, "experiments", "SOCRATES")
SOCRATES_data_dir = joinpath(CEDMF_data_dir, "experiments", "SOCRATES")


last_calibration_vars = Dict(
    ("SOCRATES_Base", "tau_autoconv_noneq") => ["ql_all_mean", "qi_all_mean"],
    ("SOCRATES_Base", "pow_icenuc_autoconv_eq") => ["temperature_mean", "ql_mean", "qi_mean"],
    ("SOCRATES_exponential_T_scaling_ice", "tau_autoconv_noneq") => ["ql_mean", "qi_mean"],
    # ("SOCRATES_exponential_T_scaling_ice_raw", "tau_autoconv_noneq") => ["ql_mean","qi_mean"],
    ("SOCRATES_powerlaw_T_scaling_ice", "tau_autoconv_noneq") => ["ql_mean", "qi_mean"],
    ("SOCRATES_geometric_liq__geometric_ice", "tau_autoconv_noneq") => ["temperature_mean", "ql_mean", "qi_mean"],
    ("SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice", "tau_autoconv_noneq") =>
        ["ql_mean", "qi_mean"],
    ("SOCRATES_geometric_liq__powerlaw_T_scaling_ice", "tau_autoconv_noneq") => ["ql_all_mean", "qi_all_mean"],
    ("SOCRATES_neural_network", "tau_autoconv_noneq") => ["temperature_mean", "ql_mean", "qi_mean"],
    ("SOCRATES_linear_combination", "tau_autoconv_noneq") => ["ql_all_mean", "qi_all_mean"],
    ("SOCRATES_linear_combination_with_w", "tau_autoconv_noneq") => ["ql_mean", "qi_mean"],
)


new_calibration_vars_list = (
    ["ql_mean", "qi_mean"],
    ["ql_all_mean", "qi_all_mean"],
    ["ql_mean", "qi_mean", "qr_mean", "qip_mean"],
    ["temperature_mean", "ql_mean", "qi_mean"],
    ["temperature_mean", "ql_all_mean", "qi_all_mean"],
)

ens_param_factors = Dict("SOCRATES_neural_network" => 1.0)


dt_min_list = [0.5, 2.0, 5.0] # factor of 10 increase is ~factor 10 decrease in number of vertical points.... we'll have to see how that looks.....
dt_max_list = dt_min_list .* 4.0 # it appears that adapt_dt isn't that smart after all, consider reducing to 2.0
adapt_dt_list = repeat([true], length(dt_min_list))

# ========================================================================================================================= #
# ==  update old calibration vars folder to have the calibration vars string in it (should be deprecated? well used again for dt) == #

fix_old_calibration_vars_dirs = false
if fix_old_calibration_vars_dirs
    for experiment in experiments
        for setup in setups
            if (experiment, setup) in keys(last_calibration_vars) # the valid ones

                for calibratation_vars in new_calibration_vars_list
                    calibration_vars_str = join(sort(calibratation_vars), "__")
                    @info "New calibration vars: $calibratation_vars"

                    dt_min = 0.5
                    dt_max = 2.0
                    adapt_dt = true
                    dt_string =
                        adapt_dt ? "adapt_dt__dt_min_" * string(dt_min) * "__dt_max_" * string(dt_max) :
                        "dt_" * string(dt_min)
                    @info(dt_string)


                    @info "Experiment: $experiment, Setup: $setup, Calibration vars: $calibration_vars_str, dt_string: $dt_string"

                    # Fix folders

                    fix_folders = true
                    if fix_folders

                        old_path = joinpath(
                            SOCRATES_dir,
                            "subexperiments",
                            experiment,
                            "Calibrate_and_Run",
                            setup,
                            calibration_vars_str,
                        )
                        new_path = joinpath(
                            SOCRATES_dir,
                            "subexperiments",
                            experiment,
                            "Calibrate_and_Run",
                            setup,
                            dt_string,
                            calibration_vars_str,
                        )

                        # make new directory
                        mkpath(new_path)

                        # mv old to new recursively
                        if isdir(joinpath(old_path, "calibrate"))
                            @info "Moving:\n $old_path/calibrate \n to:\n $new_path"
                            run(`mv $old_path/calibrate $new_path`)
                        end
                        if isdir(joinpath(old_path, "run"))
                            @info "Moving:\n $old_path/run \n to:\n $new_path"
                            run(`mv $old_path/run $new_path`)
                        end

                        sleep(0.1)

                        # move ouptput symlinks
                        # change directory to new path

                        for dir in ["calibrate", "run", "postprocessing"]

                            if isdir(joinpath(old_path, dir))
                                @info "Moving:\n $old_path/$dir \n to:\n $new_path"
                                run(`mv $old_path/$dir $new_path`)
                            end

                            run(`cd $new_path/$dir`)
                            cd(joinpath(new_path, dir))
                            rm("$new_path/$dir/output", force = true)
                            run(
                                `ln -s ../../../../../Data_Storage/Calibrate_and_Run/$setup/$dt_string/$calibration_vars_str/$dir/output  $new_path/$dir/output`,
                            )

                            if isdir(old_path)
                                @info "Removing old path: $old_path"
                                run(`rmdir $old_path`)
                            end
                        end
                    end

                    # Fix data folders
                    fix_data_folders = true
                    if fix_data_folders
                        old_path = joinpath(
                            SOCRATES_data_dir,
                            "subexperiments",
                            experiment,
                            "Calibrate_and_Run",
                            setup,
                            calibration_vars_str,
                        )
                        new_path = joinpath(
                            SOCRATES_data_dir,
                            "subexperiments",
                            experiment,
                            "Calibrate_and_Run",
                            setup,
                            dt_string,
                            calibration_vars_str,
                        )

                        # make new directory
                        mkpath(new_path)

                        # mv old to new recursively

                        for dir in ["calibrate", "run", "postprocessing"]
                            if isdir(joinpath(old_path, dir))
                                @info "Moving:\n $old_path/$dir \n to:\n $new_path"
                                run(`mv $old_path/$dir $new_path`)
                            end

                        end
                    end
                end
            end
        end
    end
end



# ========================================================================================================================= #
# == copy from one calibration variable folder to create new calibration variable folders == #

# >>> look into pulling from SOCARATE_template folders i created <<< #

add_new_calibration_vars = false # new dt too
if add_new_calibration_vars
    for experiment in experiments
        for setup in setups
            if (experiment, setup) in keys(last_calibration_vars) # the valid ones
                old_calibration_vars = last_calibration_vars[(experiment, setup)]
                old_dt_min, old_dt_max, old_adapt_dt = 0.5, 2.0, true
                old_dt_string =
                    old_adapt_dt ? "adapt_dt__dt_min_" * string(old_dt_min) * "__dt_max_" * string(old_dt_max) :
                    "dt_" * string(old_dt_min)
                old_calibration_vars_str = join(sort(old_calibration_vars), "__")

                for (dt_min, dt_max, adapt_dt) in zip(dt_min_list, dt_max_list, adapt_dt_list)
                    dt_string =
                        adapt_dt ? "adapt_dt__dt_min_" * string(dt_min) * "__dt_max_" * string(dt_max) :
                        "dt_" * string(dt_min)

                    for new_calibratation_vars in new_calibration_vars_list
                        new_calibration_vars_str = join(sort(new_calibratation_vars), "__")
                        @info "New calibration vars: $new_calibratation_vars"

                        old_path = joinpath(
                            SOCRATES_dir,
                            "subexperiments",
                            experiment,
                            "Calibrate_and_Run",
                            setup,
                            old_dt_string,
                            old_calibration_vars_str,
                        )
                        new_path = joinpath(
                            SOCRATES_dir,
                            "subexperiments",
                            experiment,
                            "Calibrate_and_Run",
                            setup,
                            dt_string,
                            new_calibration_vars_str,
                        )

                        # make new directory
                        if !isdir(new_path)
                            if !isdir(dirname(new_path))
                                mkpath(dirname(new_path))
                            end
                            run(`cp -rf $old_path $new_path`)
                        end

                        # == Copy data folders == #
                        old_data_path = joinpath(
                            SOCRATES_data_dir,
                            "subexperiments",
                            experiment,
                            "Calibrate_and_Run",
                            setup,
                            old_dt_string,
                            old_calibration_vars_str,
                        )
                        new_data_path = joinpath(
                            SOCRATES_data_dir,
                            "subexperiments",
                            experiment,
                            "Calibrate_and_Run",
                            setup,
                            dt_string,
                            new_calibration_vars_str,
                        )
                        # make new directory
                        mkpath(new_data_path)

                        if !isdir(new_data_path)
                            @info "Copying:\n $old_data_path \n to:\n $new_data_path"
                            run(`cp -rf $old_data_path $new_data_path`)
                        end

                        config_path = joinpath(new_path, "calibrate", "configs")
                        # get all files with config*.jl
                        config_files = filter(x -> occursin(r"config.*\.jl", x), readdir(config_path))
                        @info "config_files: $config_files"

                        # find the line w/ calibration_vars = [...] and replace it with calibration_vars = $new_calibratation_vars
                        for config_file in config_files
                            config_file_fullpath = joinpath(config_path, config_file)
                            for line in eachline(config_file_fullpath)
                                if occursin(r"^calibration_vars *=", line) # ^ is start of line, * matches 0 or more of the preceding character (`space` here)
                                    @info "Found line: $line"
                                    new_line = "calibration_vars = $new_calibratation_vars"
                                    @info "Replacing with: $new_line"
                                    run(`sed -i 's/calibration_vars = .*'/$new_line/ $config_file_fullpath`) # replace calibration var sline
                                    run(`sed -i 's/calibration_vars=.*'/$new_line/ $config_file_fullpath`) # replace calibration var sline
                                end

                                if occursin(r"# *calibration_vars", line) #|| occursin(r"#calibration_vars", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                    @info "Found commented line: $line"
                                    run(`sed -i '/# calibration_vars/d' $config_file_fullpath`)
                                end


                            end
                        end

                        @info("setup: ", (experiment, setup, new_calibration_vars_str))

                        # move output symlinks
                        # change directory to new path

                        for dir in ["calibrate", "run", "postprocessing"]
                            if !isdir(joinpath(new_path, dir))
                                mkpath(joinpath(new_path, dir))
                            end
                            run(`cd $new_path/$dir`)
                            cd(joinpath(new_path, dir))
                            rm("$new_path/$dir/output", force = true) # remove symlink if it exists
                            outpath = abspath(
                                joinpath(
                                    "$new_path/$dir",
                                    "../../../../../Data_Storage/Calibrate_and_Run/$setup/$dt_string/$new_calibration_vars_str/$dir/output",
                                ),
                            ) # create folder at outpath that in Data_Storage that we will link to if it doesnt already exist
                            if !isdir(outpath)
                                mkpath(outpath)
                            end
                            run(
                                `ln -s ../../../../../Data_Storage/Calibrate_and_Run/$setup/$dt_string/$new_calibration_vars_str/$dir/output  $new_path/$dir/output`,
                            )
                            run(`rm -rf ../../../../Data_Storage`)

                        end
                    end
                end
            end
        end
    end
end



# ========================================================================================================================= #
# == edit existing config files (should be deprecated by template...) == #

# edit_config_files = false
# if edit_config_files
#     for experiment in experiments
#         for setup in setups
#             if (experiment, setup) in keys(last_calibration_vars) # the valid ones

#                 for new_calibratation_vars in new_calibration_vars_list
#                     new_calibration_vars_str = join(sort(new_calibratation_vars), "__")
#                     @info "New calibration vars: $new_calibratation_vars"

#                     new_path = joinpath(SOCRATES_dir, "subexperiments", experiment, "Calibrate_and_Run", setup, new_calibration_vars_str)

#                     config_path = joinpath(new_path, "calibrate", "configs")
#                     # get all files with config*.jl
#                     config_files = filter(x -> occursin(r"config.*\.jl", x), readdir(config_path))
#                     @info "config_files: $config_files"

#                     # find the line w/ calibration_vars = [...] and replace it with calibration_vars = $new_calibratation_vars
#                     for config_file in config_files
#                         config_file_fullpath = joinpath(config_path, config_file)
#                         for line in eachline(config_file_fullpath)
#                             # if occursin(r"^calibration_vars =", line) || occursin(r"^calibration_vars=", line) # ^ is start of line
#                             #     @info "Found line: $line"
#                             #     new_line = "calibration_vars = $new_calibratation_vars"
#                             #     @info "Replacing with: $new_line"
#                             #     run(`sed -i 's/calibration_vars = .*'/$new_line/ $config_file_fullpath`) # replace calibration var sline
#                             #     run(`sed -i 's/calibration_vars=.*'/$new_line/ $config_file_fullpath`) # replace calibration var sline
#                             # end

#                             # if occursin(r"# calibration_vars", line) || occursin(r"#calibration_vars", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
#                             #     @info "Found line: $line"
#                             #     run(`sed -i '/# calibration_vars/d' $config_file_fullpath`)
#                             # end

#                             if occursin(r"^t_max =", line) || occursin(r"^t_max=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
#                                 @info "Found line: $line"
#                                 new_line = "t_max = 7*3600.0 # shorter for testing (remember to change t_start, t_end, Σ_t_start, Σ_t_end in get_reference_config()# shorter for testing" # CHANGE TO 14 LATER
#                                 # new_line = "t_max = 12*3600.0 # gametime" #

#                                 @info "Replacing with: $new_line"
#                                 run(`sed -i 's/^t_max *=.*'/$new_line/ $config_file_fullpath`)
#                                 run(`sed -i 's/^t_max=.*'/$new_line/ $config_file_fullpath`)
#                             end

#                             if occursin(r"^t_bnds =", line) || occursin(r"^t_bnds=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
#                                 @info "Found line: $line"
#                                 new_line = "t_bnds = (;obs_data = (t_max-2*3600.    , t_max), ERA5_data = (t_max-2*3600.     , t_max)) # shorter for testing" # remove this line later (to use reference period from paper)
#                                 # new_line = "t_bnds = (;obs_data = missing, ERA5_data = missing) # shorter for testing" # gametime

#                                 @info "Replacing with: $new_line"
#                                 run(`sed -i 's/^t_bnds *=.*'/$new_line/ $config_file_fullpath`)
#                                 run(`sed -i 's/^t_bnds=.*'/$new_line/ $config_file_fullpath`)
#                             end


#                         end
#                     end
#                 end
#             end
#         end
#     end
# end


# ========================================================================================================================= #
# == Copy from template to new config files == #

setup_new_scripts = true
if setup_new_scripts
    template_file =
        joinpath(SOCRATES_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_body.jl")

    calibrate_to = "Atlas_LES" # "Atlas_LES" or "Flight_Observations"
    flight_numbers = [1, 9, 10, 11, 12, 13]
    forcing_types = [:obs_data]

    N_ens = 50 # number of ensemble members (needed)
    ens_param_factor_default = 4 # scale the ensemble size by the # of parametesr we're calibrating
    use_ens_param_factor = true
    N_iter = 10 # number of iterations


    for experiment in experiments
        supersat_type = split(experiment, "_"; limit = 2)[2]
        for setup in setups
            if (experiment, setup) in keys(last_calibration_vars) # the valid ones

                for (dt_min, dt_max, adapt_dt) in zip(dt_min_list, dt_max_list, adapt_dt_list)
                    dt_string =
                        adapt_dt ? "adapt_dt__dt_min_" * string(dt_min) * "__dt_max_" * string(dt_max) :
                        "dt_" * string(dt_min)

                    for new_calibratation_vars in new_calibration_vars_list
                        new_calibration_vars_str = join(sort(new_calibratation_vars), "__")
                        @info "New calibration vars: $new_calibratation_vars"

                        new_path = joinpath(
                            SOCRATES_dir,
                            "subexperiments",
                            experiment,
                            "Calibrate_and_Run",
                            setup,
                            dt_string,
                            new_calibration_vars_str,
                        )

                        config_path = joinpath(new_path, "calibrate", "configs")
                        new_config_file = joinpath(config_path, "config_calibrate_RFAll_obs.jl")
                        # old_config_file = joinpath(config_path, "config_calibrate_RFAll_obs_old.jl")

                        test_file = joinpath(config_path, "config_calibrate_RFAll_obs_test.jl")


                        # define these as global so we can use them...
                        global calibrate_to
                        global flight_numbers
                        global forcing_types

                        if use_ens_param_factor
                            ens_param_factor = get(ens_param_factors, experiment, ens_param_factor_default)

                            read_config_way = false
                            if read_config_way # read directly form output of config, precise but slower
                                old_calibrate_to, old_flight_numbers, old_forcing_types =
                                    calibrate_to, flight_numbers, forcing_types
                                # we have to set N_iter inside the file bc ekp_par_calibration.sbatch greps for that to know how many jobs to start
                                # The only way to know how many jobs to start though is to know how many parameters we're calibrating which comes from the config file
                                # Nothing we set below should change the # of parameters we're calibrating so we should be good
                                redirect_stderr(devnull) do # hide all warnings and info messages
                                    # if calibrate edmf not loaded
                                    if !isdefined(Main, :CalibrateEDMF)
                                        Pkg.activate(CEDMF_dir)
                                    end
                                    include(new_config_file) # would set N_ens, N_iter and everything else... but doesnt bc of scoping
                                    config = @invokelatest get_process_config() # function definitions maybe should work?
                                end
                                calibrate_to, flight_numbers, forcing_types =
                                    old_calibrate_to, old_flight_numbers, old_forcing_types

                            else # try to get calibration_parametesr from config_calibrate_template_header and calibration_parameters_and_nameslit

                                for line in eachline(new_config_file)
                                    if occursin(r"^header_setup_choice\s+=", line)
                                        @info("Found line: $line")
                                        println(line)
                                        # header_setup_choice = parse(Symbol,  split(line, ('=', '#'))[2]) # should really be a split at comment character '#' like above  in case comment includes digits
                                        headers_setup_choice = eval(Meta.parse(split(line, ('=', '#'))[2]))
                                    end
                                end

                                if !isdefined(Main, :simple_calibration_parameters) ||
                                   !isdefined(Main, :default_calibration_parameters)
                                    include(
                                        joinpath(
                                            SOCRATES_dir,
                                            "Calibrate_and_Run_scripts",
                                            "calibrate",
                                            "config_calibrate_template_header.jl",
                                        ),
                                    )
                                end

                                if header_setup_choice == :simple
                                    calibration_parameters = simple_calibration_parameters
                                elseif header_setup_choice == :default
                                    calibration_parameters = default_calibration_parameters
                                else
                                    error("header_setup_choice not recognized")
                                end

                                @info(
                                    "loading:",
                                    joinpath(
                                        SOCRATES_dir,
                                        "subexperiments",
                                        experiment,
                                        "Calibrate_and_Run",
                                        setup,
                                        "calibration_parameters_and_namelist.jl",
                                    )
                                )
                                flush(stdout)
                                flush(stderr)

                                global experiment_dir =
                                    joinpath(CEDMF_dir, "experiments", "SOCRATES", "subexperiments", experiment) # used for NN but only defined in config_calibrate_template_body.jl since it's defined after supersat_type
                                global calibration_setup = setup # it's called that in calibration_parameters_and_namelist.jl files
                                include(
                                    joinpath(
                                        SOCRATES_dir,
                                        "subexperiments",
                                        experiment,
                                        "Calibrate_and_Run",
                                        setup,
                                        "calibration_parameters_and_namelist.jl",
                                    ),
                                )

                                calibration_parameters =
                                    merge(calibration_parameters, calibration_parameters__experiment_setup)


                            end
                            N_params = sum([length(v["prior_mean"]) for v in values(calibration_parameters)])
                            global N_ens = Int(ceil(N_params * ens_param_factor)) + 1
                            @info("N_ens: $N_ens")
                        end

                        run(`rm -rf $test_file`)

                        # copy template file to new config file
                        # run(`rm -rf $old_config_file`)
                        # run(`cp $new_config_file $old_config_file`)
                        run(`cp $template_file $new_config_file`)

                        # fix up the new config file
                        for line in eachline(new_config_file)


                            # ========================================================================================================================= #

                            if occursin(r"^supersat_type *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "supersat_type = :$supersat_type"

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^supersat_type *=.*'/$new_line/ $new_config_file`)
                            end

                            if occursin(r"^calibration_setup *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "calibration_setup = \"$setup\"" # remove this line later (to use reference period from paper)

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^calibration_setup *=.*'/$new_line/ $new_config_file`)
                            end

                            if occursin(r"^calibrate_to *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "calibrate_to = \"$calibrate_to\"" # remove this line later (to use reference period from paper)

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^calibrate_to *=.*'/$new_line/ $new_config_file`)
                            end

                            if occursin(r"^flight_numbers *=", line)  # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "flight_numbers = Vector{Int}($flight_numbers)" # remove this line later (to use reference period from paper)

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^flight_numbers *=.*'/$new_line/ $new_config_file`)
                            end

                            if occursin(r"^forcing_types *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "forcing_types = Vector{Symbol}($forcing_types)" # remove this line later (to use reference period from paper)

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^forcing_types *=.*'/$new_line/ $new_config_file`)
                            end

                            if occursin(r"^calibration_vars *=", line) # ^ is start of line
                                @info "Found line: $line"
                                new_line = "calibration_vars = $new_calibratation_vars"
                                @info "Replacing with: $new_line"
                                run(`sed -i 's/calibration_vars *=.*'/$new_line/ $new_config_file`) # replace calibration var sline
                            end

                            # ========================================================================================================================= #

                            if occursin(r"^t_max *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "t_max = 7*3600.0 # shorter for testing (remember to change t_start, t_end, Σ_t_start, Σ_t_end in get_reference_config()# shorter for testing" # CHANGE TO 14 LATER
                                # new_line = "t_max = 12*3600.0 # gametime" #

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^t_max *=.*'/$new_line/ $new_config_file`)
                            end

                            if occursin(r"^t_bnds *=", line)# delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "t_bnds = (;obs_data = (t_max-2*3600.    , t_max), ERA5_data = (t_max-2*3600.     , t_max)) # shorter for testing" # remove this line later (to use reference period from paper)
                                # new_line = "t_bnds = (;obs_data = missing, ERA5_data = missing) # shorter for testing" # gametime

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^t_bnds *=.*'/$new_line/ $new_config_file`)
                            end

                            if occursin(r"^dt_min *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "dt_min = $dt_min"

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^dt_min *=.*'/$new_line/ $new_config_file`)
                            end

                            if occursin(r"^dt_max *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "dt_max = $dt_max"

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^dt_max *=.*'/$new_line/ $new_config_file`)
                            end

                            if occursin(r"^adapt_dt *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "adapt_dt = $adapt_dt"

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^adapt_dt *=.*'/$new_line/ $new_config_file`)
                            end

                            # ========================================================================================================================= #

                            if occursin(r"^N_ens *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "N_ens = $N_ens" # remove this line later (to use reference period from paper)

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^N_ens *=.*'/$new_line/ $new_config_file`)
                            end

                            # if occursin(r"^ens_param_factor *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            #     @info "Found line: $line"
                            #     new_line = "ens_param_factor = $ens_param_factor" # remove this line later (to use reference period from paper)

                            #     @info "Replacing with: $new_line"
                            #     run(`sed -i 's/^ens_param_factor *=.*'/$new_line/ $new_config_file`)
                            # end

                            # if occursin(r"^use_ens_param_factor *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            #     @info "Found line: $line"
                            #     new_line = "use_ens_param_factor = $use_ens_param_factor" # remove this line later (to use reference period from paper)

                            #     @info "Replacing with: $new_line"
                            #     run(`sed -i 's/^use_ens_param_factor *=.*'/$new_line/ $new_config_file`)
                            # end


                            if occursin(r"^N_iter *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found line: $line"
                                new_line = "N_iter = $N_iter" # remove this line later (to use reference period from paper)

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^N_iter *=.*'/$new_line/ $new_config_file`)
                            end



                            # ========================================================================================================================= #


                        end
                    end
                end
            end
        end
    end
end







# A way to create calibration vars scripts by folder (replace that line automatically? idk)




# Go edit the python code for reading to add these variable comos as an organization level in the caliration_data structure... (is just like setup i guess? idk...)
