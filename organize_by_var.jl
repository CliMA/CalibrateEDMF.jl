
# experiments list
experiments=(
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

setups=("pow_icenuc_autoconv_eq", "tau_autoconv_noneq")

  

CEDMF_dir = "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl"
CEDMF_data_dir = "/home/jbenjami/Data/Research_Schneider/CliMa/CalibrateEDMF.jl/"

SOCRATES_dir = joinpath(CEDMF_dir, "experiments", "SOCRATES")
SOCRATES_data_dir = joinpath(CEDMF_data_dir, "experiments", "SOCRATES")


last_calibration_vars = Dict(
    ("SOCRATES_Base", "tau_autoconv_noneq") => ["ql_all_mean","qi_all_mean"],
    ("SOCRATES_Base", "pow_icenuc_autoconv_eq") => ["temperature_mean", "ql_mean", "qi_mean"],
    ("SOCRATES_exponential_T_scaling_ice", "tau_autoconv_noneq") => ["ql_mean", "qi_mean"],
    ("SOCRATES_exponential_T_scaling_ice_raw", "tau_autoconv_noneq") => ["ql_mean","qi_mean"],
    ("SOCRATES_powerlaw_T_scaling_ice", "tau_autoconv_noneq") => ["ql_mean","qi_mean"],
    ("SOCRATES_geometric_liq__geometric_ice", "tau_autoconv_noneq") => ["temperature_mean", "ql_mean", "qi_mean"],
    ("SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice", "tau_autoconv_noneq") => ["ql_mean", "qi_mean"],
    ("SOCRATES_geometric_liq__powerlaw_T_scaling_ice", "tau_autoconv_noneq") => ["ql_all_mean", "qi_all_mean"],
    ("SOCRATES_neural_network", "tau_autoconv_noneq") => ["temperature_mean", "ql_mean", "qi_mean"],
    ("SOCRATES_linear_combination", "tau_autoconv_noneq") => ["ql_all_mean", "qi_all_mean"],
    ("SOCRATES_linear_combination_with_w", "tau_autoconv_noneq") => ["ql_mean","qi_mean"],
)


new_calibration_vars_list = (
    ["ql_mean", "qi_mean"],
    ["ql_all_mean", "qi_all_mean"],
    ["ql_mean", "qi_mean", "qr_mean", "qip_mean"],
    ["temperature_mean", "ql_mean", "qi_mean"],
    ["temperature_mean", "ql_all_mean", "qi_all_mean"],
)

fix_old_calibration_vars_dirs = false
if fix_old_calibration_vars_dirs
    for experiment in experiments
        for setup in setups
            if (experiment, setup) in keys(last_calibration_vars)
                calibration_vars = last_calibration_vars[(experiment, setup)]
                calibration_vars_str = join(sort(calibration_vars), "__")

                @info "Experiment: $experiment, Setup: $setup, Calibration vars: $calibration_vars_str"

                # Fix folders

                fix_folders = true
                if fix_folders

                    old_path = joinpath(SOCRATES_dir, "subexperiments", experiment, "Calibrate_and_Run", setup)
                    new_path = joinpath(SOCRATES_dir, "subexperiments", experiment, "Calibrate_and_Run", setup, calibration_vars_str)

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


                    # move ouptput symlinks
                    # change directory to new path
                    run(`cd $new_path/calibrate`);cd(joinpath(new_path, "calibrate"))
                    rm("$new_path/calibrate/output", force=true)
                    run(`ln -s ../../../../Data_Storage/Calibrate_and_Run/$setup/$calibration_vars_str/calibrate/output  $new_path/calibrate/output`)

                    run(`cd $new_path/run`); cd(joinpath(new_path, "run"))
                    rm("$new_path/run/output", force=true)
                    run(`ln -s ../../../../Data_Storage/Calibrate_and_Run/$setup/$calibration_vars_str/run/output $new_path/run/output`)
                end

                # Fix data folders
                fix_data_folders = false
                if fix_data_folders
                    old_path = joinpath(SOCRATES_data_dir, "subexperiments", experiment, "Calibrate_and_Run", setup)
                    new_path = joinpath(SOCRATES_data_dir, "subexperiments", experiment, "Calibrate_and_Run", setup, calibration_vars_str)

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
                end
            end
        end
    end
end




add_new_calibration_vars = false
if add_new_calibration_vars
    for experiment in experiments
        for setup in setups
            if (experiment, setup) in keys(last_calibration_vars) # the valid ones
                old_calibration_vars = last_calibration_vars[(experiment, setup)]
                old_calibration_vars_str = join(sort(old_calibration_vars), "__")

                for new_calibratation_vars in new_calibration_vars_list
                    new_calibration_vars_str = join(sort(new_calibratation_vars), "__")
                    @info "New calibration vars: $new_calibratation_vars"

                    old_path = joinpath(SOCRATES_dir, "subexperiments", experiment, "Calibrate_and_Run", setup, old_calibration_vars_str)
                    new_path = joinpath(SOCRATES_dir, "subexperiments", experiment, "Calibrate_and_Run", setup, new_calibration_vars_str)

                    # make new directory
                    if !isdir(new_path)
                        run(`cp -rf $old_path $new_path`)
                    end

                    # == Copy data folders == #
                    old_data_path = joinpath(SOCRATES_data_dir, "subexperiments", experiment, "Calibrate_and_Run", setup, old_calibration_vars_str)
                    new_data_path = joinpath(SOCRATES_data_dir, "subexperiments", experiment, "Calibrate_and_Run", setup, new_calibration_vars_str)
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
                            if occursin(r"^calibration_vars =", line) || occursin(r"^calibration_vars=", line) # ^ is start of line
                                @info "Found line: $line"
                                new_line = "calibration_vars = $new_calibratation_vars"
                                @info "Replacing with: $new_line"
                                run(`sed -i 's/calibration_vars = .*'/$new_line/ $config_file_fullpath`) # replace calibration var sline
                                run(`sed -i 's/calibration_vars=.*'/$new_line/ $config_file_fullpath`) # replace calibration var sline
                            end

                            if occursin(r"# calibration_vars", line) || occursin(r"#calibration_vars", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found commented line: $line"
                                run(`sed -i '/# calibration_vars/d' $config_file_fullpath`)
                            end

                            if occursin(r"#^t_max =", line) || occursin(r"#^t_max=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                new_line = "t_max = 7*3600.0 # shorter for testing (remember to change t_start, t_end, Σ_t_start, Σ_t_end in get_reference_config()# shorter for testing" # CHANGE TO 14 LATER
                                @info "Replacing with: $new_line"
                                run(`sed -i 's/#^t_max *=.*'/$new_line/ $config_file_fullpath`)
                                run(`sed -i 's/#^t_max=.*'/$new_line/ $config_file_fullpath`)
                            end

                            if occursin(r"#^t_bnds =", line) || occursin(r"#^t_bnds=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                new_line = "t_bnds =  (t_max-2*3600.    , t_max), ERA5_data = (t_max-2*3600.     , t_max)) # shorter for testing" # remove this line later (to use reference period from paper)
                                @info "Replacing with: $new_line"
                                run(`sed -i 's/#^t_bnds *=.*'/$new_line/ $config_file_fullpath`)
                                run(`sed -i 's/#^t_bnds=.*'/$new_line/ $config_file_fullpath`)
                            end
                            

                        end
                    end




                    # move output symlinks
                    # change directory to new path
                    run(`cd $new_path/calibrate`); cd(joinpath(new_path, "calibrate"))
                    rm("$new_path/calibrate/output", force=true) # remove symlink if it exists
                    
                    # calibrate dir
                    outpath = abspath(joinpath( "$new_path/calibrate", "../../../../Data_Storage/Calibrate_and_Run/$setup/$new_calibration_vars_str/calibrate/output")) # create folder at outpath that in Data_Storage that we will link to if it doesnt already exist
                    if !isdir(outpath)
                        mkpath(outpath)
                    end
                    run(`ln -s ../../../../Data_Storage/Calibrate_and_Run/$setup/$new_calibration_vars_str/calibrate/output  $new_path/calibrate/output`)

                    # run dir
                    run(`cd $new_path/run`); cd(joinpath(new_path, "run"))
                    rm("$new_path/run/output", force=true) # remove symlink if it exists
                    outpath = abspath(joinpath( "$new_path/run", "../../../../Data_Storage/Calibrate_and_Run/$setup/$new_calibration_vars_str/run/output")) # create folder at outpath that in Data_Storage that we will link to if it doesnt already exist
                    if !isdir(outpath)
                        mkpath(outpath)
                    end
                    run(`ln -s ../../../../Data_Storage/Calibrate_and_Run/$setup/$new_calibration_vars_str/run/output $new_path/run/output`)
                end
            end
        end
    end
end



edit_config_files = false
if edit_config_files
    for experiment in experiments
        for setup in setups
            if (experiment, setup) in keys(last_calibration_vars) # the valid ones

                for new_calibratation_vars in new_calibration_vars_list
                    new_calibration_vars_str = join(sort(new_calibratation_vars), "__")
                    @info "New calibration vars: $new_calibratation_vars"

                    new_path = joinpath(SOCRATES_dir, "subexperiments", experiment, "Calibrate_and_Run", setup, new_calibration_vars_str)

                    config_path = joinpath(new_path, "calibrate", "configs")
                    # get all files with config*.jl
                    config_files = filter(x -> occursin(r"config.*\.jl", x), readdir(config_path))
                    @info "config_files: $config_files"

                    # find the line w/ calibration_vars = [...] and replace it with calibration_vars = $new_calibratation_vars
                    for config_file in config_files
                        config_file_fullpath = joinpath(config_path, config_file)
                        for line in eachline(config_file_fullpath)
                            # if occursin(r"^calibration_vars =", line) || occursin(r"^calibration_vars=", line) # ^ is start of line
                            #     @info "Found line: $line"
                            #     new_line = "calibration_vars = $new_calibratation_vars"
                            #     @info "Replacing with: $new_line"
                            #     run(`sed -i 's/calibration_vars = .*'/$new_line/ $config_file_fullpath`) # replace calibration var sline
                            #     run(`sed -i 's/calibration_vars=.*'/$new_line/ $config_file_fullpath`) # replace calibration var sline
                            # end

                            # if occursin(r"# calibration_vars", line) || occursin(r"#calibration_vars", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            #     @info "Found commented line: $line"
                            #     run(`sed -i '/# calibration_vars/d' $config_file_fullpath`)
                            # end

                            if occursin(r"^t_max =", line) || occursin(r"^t_max=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found commented line: $line"
                                new_line = "t_max = 7*3600.0 # shorter for testing (remember to change t_start, t_end, Σ_t_start, Σ_t_end in get_reference_config()# shorter for testing" # CHANGE TO 14 LATER
                                # new_line = "t_max = 12*3600.0 # gametime" #

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^t_max *=.*'/$new_line/ $config_file_fullpath`)
                                run(`sed -i 's/^t_max=.*'/$new_line/ $config_file_fullpath`)
                            end

                            if occursin(r"^t_bnds =", line) || occursin(r"^t_bnds=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                                @info "Found commented line: $line"
                                new_line = "t_bnds = (;obs_data = (t_max-2*3600.    , t_max), ERA5_data = (t_max-2*3600.     , t_max)) # shorter for testing" # remove this line later (to use reference period from paper)
                                # new_line = "t_bnds = (;obs_data = missing, ERA5_data = missing) # shorter for testing" # gametime

                                @info "Replacing with: $new_line"
                                run(`sed -i 's/^t_bnds *=.*'/$new_line/ $config_file_fullpath`)
                                run(`sed -i 's/^t_bnds=.*'/$new_line/ $config_file_fullpath`)
                            end
                            

                        end
                    end
                end
            end
        end
    end
end



setup_new_scripts = true
if setup_new_scripts
    template_file = joinpath(SOCRATES_dir, "Calibrate_and_Run_scripts", "calibrate", "config_calibrate_template_body.jl")
    calibrate_to = "Atlas_LES" # "Atlas_LES" or "Flight_Observations"
    flight_numbers = [1,9,10,11,12,13]
    forcing_types  = [:obs_data]
    N_ens  = 100 # number of ensemble members (needed)
    N_iter = 20 # number of iterations

    for experiment in experiments
        supersat_type = split(experiment, "_"; limit = 2)[2]
        for setup in setups
            if (experiment, setup) in keys(last_calibration_vars) # the valid ones

                for new_calibratation_vars in new_calibration_vars_list
                    new_calibration_vars_str = join(sort(new_calibratation_vars), "__")
                    @info "New calibration vars: $new_calibratation_vars"

                    new_path = joinpath(SOCRATES_dir, "subexperiments", experiment, "Calibrate_and_Run", setup, new_calibration_vars_str)

                    config_path = joinpath(new_path, "calibrate", "configs")
                    new_config_file = joinpath(config_path, "config_calibrate_RFAll_obs.jl")
                    old_config_file = joinpath(config_path, "config_calibrate_RFAll_obs_old.jl")

                    test_file = joinpath(config_path, "config_calibrate_RFAll_obs_test.jl")

                    run(`rm -rf $test_file`)

                    # copy template file to new config file
                    run(`rm -rf $old_config_file`)
                    # run(`cp $new_config_file $old_config_file`)
                    run(`cp $template_file $new_config_file`)

                    # fix up the new config file
                    for line in eachline(new_config_file)
                        

                        # ========================================================================================================================= #

                        if occursin(r"^supersat_type *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            @info "Found commented line: $line"
                            new_line = "supersat_type = :$supersat_type"

                            @info "Replacing with: $new_line"
                            run(`sed -i 's/^supersat_type *=.*'/$new_line/ $new_config_file`)
                        end

                        if occursin(r"^calibration_setup *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            @info "Found commented line: $line"
                            new_line = "calibration_setup = \"$setup\"" # remove this line later (to use reference period from paper)

                            @info "Replacing with: $new_line"
                            run(`sed -i 's/^calibration_setup *=.*'/$new_line/ $new_config_file`)
                        end

                        if occursin(r"^calibrate_to *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            @info "Found commented line: $line"
                            new_line = "calibrate_to = \"$calibrate_to\"" # remove this line later (to use reference period from paper)

                            @info "Replacing with: $new_line"
                            run(`sed -i 's/^calibrate_to *=.*'/$new_line/ $new_config_file`)
                        end

                        if occursin(r"^flight_numbers *=", line)  # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            @info "Found commented line: $line"
                            new_line = "flight_numbers = Vector{Int}($flight_numbers)" # remove this line later (to use reference period from paper)

                            @info "Replacing with: $new_line"
                            run(`sed -i 's/^flight_numbers *=.*'/$new_line/ $new_config_file`)
                        end

                        if occursin(r"^forcing_types *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            @info "Found commented line: $line"
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
                            @info "Found commented line: $line"
                            new_line = "t_max = 7*3600.0 # shorter for testing (remember to change t_start, t_end, Σ_t_start, Σ_t_end in get_reference_config()# shorter for testing" # CHANGE TO 14 LATER
                            # new_line = "t_max = 12*3600.0 # gametime" #

                            @info "Replacing with: $new_line"
                            run(`sed -i 's/^t_max *=.*'/$new_line/ $new_config_file`)
                        end

                        if occursin(r"^t_bnds *=", line)# delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            @info "Found commented line: $line"
                            new_line = "t_bnds = (;obs_data = (t_max-2*3600.    , t_max), ERA5_data = (t_max-2*3600.     , t_max)) # shorter for testing" # remove this line later (to use reference period from paper)
                            # new_line = "t_bnds = (;obs_data = missing, ERA5_data = missing) # shorter for testing" # gametime

                            @info "Replacing with: $new_line"
                            run(`sed -i 's/^t_bnds *=.*'/$new_line/ $new_config_file`)
                        end

                        if occursin(r"^N_ens *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            @info "Found commented line: $line"
                            new_line = "N_ens = $N_ens" # remove this line later (to use reference period from paper)

                            @info "Replacing with: $new_line"
                            run(`sed -i 's/^N_ens *=.*'/$new_line/ $new_config_file`)
                        end

                        if occursin(r"^N_iter *=", line) # delete these now that each variable pair gets its own folder (we did this mostly so we can run them simultaneously programatically (without needing to edit files))
                            @info "Found commented line: $line"
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







# A way to create calibration vars scripts by folder (replace that line automatically? idk)




# Go edit the python code for reading to add these variable comos as an organization level in the caliration_data structure... (is just like setup i guess? idk...)