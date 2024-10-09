using CalibrateEDMF
using CalibrateEDMF.TCRunnerUtils
import CalibrateEDMF.ReferenceModels: NameList
using Dates
using JLD2
using PyCall # for pyimport


this_dir = @__DIR__ # the location of this file
pkg_dir = pkgdir(CalibrateEDMF)
experiment_dir = joinpath(pkg_dir, "experiments", "SOCRATES_Test_Dynamical_Calibration")
truth_dir = joinpath(experiment_dir, "Truth") # the folder where we store our truth (Atlas LES Data)
include(joinpath(pkg_dir, "tools", "TCRunnerUtils.jl")) # this is the file that has the run_TCs() function
this_calibrate_and_run = "tau_autoconv_noneq"
flight_str = "09"
calibrated_config_dir = "RF" * flight_str * "_obs"
calibrated_config_path =
    joinpath(experiment_dir, "Calibrate_and_Run", this_calibrate_and_run, "calibrate", "output", calibrated_config_dir)
@info("here", calibrated_config_path)

# setup what we want to run
flight_numbers = [9]
forcing_types = [:obs_data, :ERA5_data]

use_ramp = true
# if use_ramp
#     pow_icenuc = 1.0 # the default
# else
#     pow_icenuc = 1e7 # some large number
# end
user_args = (; use_ramp = true, use_supersat = true) # use_ramp is now pow_icenuc so probably could be deprecated, use_supersat essential for noneq

setups = collect(Iterators.product(flight_numbers, forcing_types))[:] # we just keep it combined right now and filter what we want later
setups = map(x -> Dict("flight_number" => x[1], "forcing_type" => x[2]), setups) # convert to list of dictionaries

is_valid_setup =
    setup -> begin  # filter out if datafile doesn't exist (11 :obsdata for example) (logger seems unable to handle failures that arent return codes i guess...)
        name = "RF" * string(setup["flight_number"], pad = 2) * "_" * string(setup["forcing_type"])
        datafile = joinpath(truth_dir, name, "stats", name * ".nc")
        isfile(datafile)
    end
setups = filter(is_valid_setup, setups) # filter out if datafile doesn't exist (11 :obsdata for example)

case_names =
    ["SOCRATES_RF" * string(setup["flight_number"], pad = 2) * "_" * string(setup["forcing_type"]) for setup in setups]

calibrated_params = get_calibrated_parameters(calibrated_config_path; method = "final_mean")
calibrated_params = calibrated_params[1] # we calibrated on two flights, but we didn't use any param_maps or anything so they're the same, so just take the first one   


t_max = [14 * 3600] # default

# need to add functionality in TrainTau.jl run_all_SOCRATES.jl
# you can have multiple parameter dicts combine into setups if they all have the same keys 
parameters_list = []
# push!(parameters_list, Dict( # eq obs
#     "case_name" =>  filter(contains("obs"),case_names),
#     "τ_sub_dep" => [0], # toml don't work as NaN, only did bc our last_nested_dict was broken, this better anyway... updated tho to just be 0, not implemented in namelist yet outside of TrainTau branch... (should work though..., just need to edit namelist directly...)
#     "τ_cond_evap" => [0],
#     "moisture_model" => ["equilibrium", ],
#     "precipitation_model" => ["clima_1m"],
#     "t_max" => t_max,
#     "sgs" => ["mean"], # sgs mean only for nonequilibrium (should we revisit this?) particularly for liquid topped clouds...
#     "uuid" => ["out",], # suffix, we use this here to stop TC.jl from generating a random number... (now we need to specify output carefully though for no clobber...)
#     "user_args" => [user_args],
#     # "pow_icenuc" => [pow_icenuc], # should get set from calibrated params... i think
# ))
# push!(parameters_list, Dict( # eq ERA5 
#     "case_name" =>  filter(contains("ERA5"),case_names),
#     "τ_sub_dep" => [0], # toml don't work as NaN, only did bc our last_nested_dict was broken, this better anyway... updated tho to just be 0, not implemented in namelist yet outside of TrainTau branch... (should work though..., just need to edit namelist directly...)
#     "τ_cond_evap" => [0],
#     "moisture_model" => ["equilibrium", ],
#     "precipitation_model" => ["clima_1m"],
#     "t_max" => t_max,
#     "sgs" => ["mean"], # sgs mean only for nonequilibrium (should we revisit this?) particularly for liquid topped clouds...
#     "uuid" => ["out",], # suffix, we use this here to stop TC.jl from generating a random number... (now we need to specify output carefully though for no clobber...)
#     "user_args" => [user_args],
#     "pow_icenuc" => [pow_icenuc], # use_ramp
# ))
push!(
    parameters_list,
    Dict( # noneq obs
        "case_name" => filter(contains("obs"), case_names),
        "τ_sub_dep" => 10 .^ range(0.3, 5, 10), # not implemented in namelist yet outside of TrainTau branch... (should work though..., just need to edit namelist directly...)
        "τ_cond_evap" => 10 .^ range(0.3, 5, 10),
        "moisture_model" => ["nonequilibrium"],
        "precipitation_model" => ["clima_1m"],
        "t_max" => t_max,
        "sgs" => ["mean"], # sgs mean only for nonequilibrium (should we revisit this?) particularly for liquid topped clouds...
        "uuid" => ["out"], # suffix, we use this here to stop TC.jl from generating a random number... (now we need to specify output carefully though for no clobber...)
        "user_args" => [user_args],
        # "pow_icenuc" => [pow_icenuc], # use_ramp
    ),
)
# push!(parameters_list, Dict( # noneq ERA5
#     "case_name" =>  filter(contains("ERA5"),case_names),
#     "τ_sub_dep" => 10 .^ range(0.3,5,10), # not implemented in namelist yet outside of TrainTau branch... (should work though..., just need to edit namelist directly...)
#     "τ_cond_evap" => 10 .^ range(0.3,5,10),
#     "moisture_model" => [ "nonequilibrium",],
#     "precipitation_model" => ["clima_1m"],
#     "t_max" => t_max,
#     "sgs" => ["mean"], # sgs mean only for nonequilibrium (should we revisit this?) particularly for liquid topped clouds...
#     "uuid" => ["out",], # suffix, we use this here to stop TC.jl from generating a random number... (now we need to specify output carefully though for no clobber...)
#     "user_args" => [user_args],
#     "pow_icenuc" => [pow_icenuc], # use_ramp
# ))


# # test
# push!(parameters_list, Dict( # noneq ERA5
#     "case_name" =>  ["SOCRATES_RF12_ERA5_data"],
#     "τ_sub_dep" => 10 .^ range(0.3,5,2), # not implemented in namelist yet outside of TrainTau branch... (should work though..., just need to edit namelist directly...)
#     "τ_cond_evap" => 10 .^ range(0.3,5,2),
#     "moisture_model" => [ "equilibrium",],
#     "precipitation_model" => ["clima_1m"],
#     "t_max" => t_max,
#     "sgs" => ["mean"], # sgs mean only for nonequilibrium (should we revisit this?) particularly for liquid topped clouds...
#     "uuid" => ["out",], # suffix, we use this here to stop TC.jl from generating a random number... (now we need to specify output carefully though for no clobber...)
#     "user_args" => [user_args],
#     "pow_icenuc" => [pow_icenuc], # use_ramp
# ))



name_heirarchy = Dict( # Copied from run_all_SOCRATES.jl in TrainTau.jl, just so works with our existing python code... not ideal stuff lol, these need to match parameters...
    0 => ["case_name", "moisture_model", "precipitation_model", "sgs", "dt", "t_max"], # case_name is case in our legacy code lol rip...
    1 => ["τ_sub_dep", "τ_cond_evap"],
)

alt_param_placements = Dict( # for those not in the default namelist, but we know where they go...
    "τ_sub_dep" => ("microphysics",),
    "τ_cond_evap" => ("microphysics",),
    "user_args" => (),
    # "pow_icenuc" => ("microphysics",),
)

parameters_output_dirs_list = Vector{String}(undef, length(parameters_list)) # this we only store one for each parameter dict, not sone for each setup
setups_output_dirs_list = Vector{Array}(undef, length(parameters_list)) # one for each setup
setups_list = Vector{Array}(undef, length(parameters_list))
out_datas = Vector{Dict}(undef, length(parameters_list))
loss_data = Vector{Dict}(undef, length(parameters_list))

sorted_setups = Vector{Array}(undef, length(parameters_list))
descriptors = Vector{Array}(undef, length(parameters_list))
names = Vector{Array}(undef, length(parameters_list))

datetime_stamp = Dates.format(Dates.now(), "yyyy_mm_dd__HHMM_SS") # save here so they all get the same time
# datetime_suffixes = string.(1: length(parameters_list))# array so we can increment it without using global or let block
datetime_suffixes = [
    string(forcing_types[contains.(parameters["case_name"][1], string.(forcing_types))][]) *
    "__" *
    parameters["moisture_model"][] for parameters in parameters_list
] # works with our current implementation

for (i, parameters) in enumerate(parameters_list) # store things so we can iterate jobs regardless of which parameters dict theyre from, so that we can keep all workings going all the time (since the parameters dicts ave different numbers of jobs)
    local_setups = collect(Iterators.product(values(parameters)...))[:] # different name to not conflict w/ global
    setups_list[i] = [Dict(keys(parameters) .=> setup) for setup in local_setups] # make these generators once it's all working?
    # output_dir = joinpath(experiment_dir, "Output", "Calibrated_Runs", datetime_stamp, datetime_stamp * "__" * datetime_suffixes[i]) # store them inside each other for organization, but inner still w/ datetime so can send to jupyter workflow easily etc...
    output_dir = joinpath(
        experiment_dir,
        "Calibrate_and_Run",
        this_calibrate_and_run,
        "run",
        "output",
        calibrated_config_dir,
        datetime_suffixes[i],
    )  # drop the datetime info, just keep the file at the suffix so if we run again it just overwrites...
    mkpath(output_dir) # make it now so we only have to do it once...
    cp(
        @__FILE__,
        joinpath(output_dir, "run_calibrated_" * this_calibrate_and_run * "_RF" * flight_str * "_obs.jl"),
        force = true,
    )
    cp(joinpath(calibrated_config_path, "config.jl"), joinpath(output_dir, "config.jl"), force = true)
    parameters_output_dirs_list[i] = output_dir # storing for post processing... since it gets flattened otherwise
    setups_output_dirs_list[i] = repeat([output_dir], length(setups_list[i])) # one for each setup here
    sleep(1) # so we don't get the same timestamp for all of them...

    # warn if will overwrite calibrated parameters
    for param in keys(parameters)
        if param in keys(calibrated_params)
            @warn("will overwrite calibrated parameter $param with specified values in parameters dict(s)")
        end
    end

    sorted_setups[i] = @. sort(collect(setups_list[i]), by = x -> x[1]) # sort by key
    descriptors[i] = [
        rstrip(
            join([
                string(key) * "-" * string(val) * "__" for
                (key, val) in filter(p -> p[1] in name_heirarchy[0], sorted_setup)
            ]),
            '_',
        ) for sorted_setup in sorted_setups[i]
    ]
    names[i] = [
        rstrip(
            join([
                string(key) * "-" * string(round(log10(val), sigdigits = 4)) * "__" for
                (key, val) in filter(p -> p[1] in name_heirarchy[1], sorted_setup)
            ]),
            '_',
        ) for sorted_setup in sorted_setups[i]
    ]# we don't have a dict w/ user_args, user_aux so this should be simpler too i think... copy same code from descriptor

    # post processing stuff
    out_datas[i] = Dict(
        "dir_array" => Array{String}(undef, (length(parameters["τ_cond_evap"]), length(parameters["τ_sub_dep"]))),
        "loss_array" => fill(NaN, (length(parameters["τ_cond_evap"]), length(parameters["τ_sub_dep"]))), # placeholder, we don't calculate any loss here, we just do that in python now...
        "tau" => Dict("liq" => parameters["τ_cond_evap"], "ice" => parameters["τ_sub_dep"]),
    ) # a dict copying format from TrainTau.jl, used to label the output data after process_sensitivity_set
    out_datas[i]["dir_array"] =
        reshape(unique(names[i]), length(parameters["τ_cond_evap"]), length(parameters["τ_sub_dep"])) # unique cause could be repeated from different cases and such... techincally though this only works if everythin else only has a single option like we did in TrainTau.jl too....

    # for (n1,τ_cond_evap) in enumerate(parameters["τ_cond_evap"])
    #     for (n2,τ_sub_dep) in enumerate(parameters["τ_sub_dep"])
    #         out_datas[i]["dir_array"][n1,n2] = strip(join([string(key)*"-"*string(round(log10(val),sigdigits=4))*"__" for (key,val) in zip(name_heirarchy[1], (τ_cond_evap, τ_sub_dep))]),'_') *"/"  # same as name below, maybe that group could be moved up here to reduce redundncy...
    #     end
    # end

end

setups_list = collect(Iterators.flatten(setups_list))
setups_output_dirs_list = collect(Iterators.flatten(setups_output_dirs_list))


@info length(setups_list)
@info length(setups_output_dirs_list)


# @info setups_list
# @info output_dirs_list


# for parameters in parameters_list # iterate over our simulations for now instead of merging them... if we merge them but want separation we could look into ways to keep their timestamps separate...

# datetime_stamp = Dates.format(Dates.now(),"yyyy_mm_dd__HHMM_SS") # save here so they all get the same time

# copy this script to the output dir
# output_dir = joinpath(experiment_dir, "Output", "Calibrated_Runs", datetime_stamp)
# mkpath(output_dir) # make the directory if it doesn't exist
# cp(@__FILE__, joinpath(output_dir, "run_calibrated_autoconv.jl"))
# cp(joinpath(calibrated_config_path, "config.jl"), joinpath(output_dir, "config.jl"))

# # warn if overwriting calibrated parameters
# for param in keys(parameters)
#     if param in keys(calibrated_params)
#         @warn("overwriting calibrated parameter $param with specified values in parameters dict")
#     end
# end

# setups = collect(Iterators.product(values(parameters)...))[:]
# setups = [Dict(keys(parameters) .=> setup) for setup in setups]

# create namelists (namelist for each case with our requested namelist params, can't use config cause it's not clear what we're running.
namelists = []

# for setup in setups
for (setup, output_dir) in zip(setups_list, setups_output_dirs_list)

    # create namelist
    @info(string(setup["case_name"]))
    namelist = NameList.default_namelist.(setup["case_name"], write = false, set_seed = false)
    # add in our specified params (not sure which of these methods I'm supposed to be using...)
    # @info(namelist)


    # add in the calibrated params (do first)
    for (pName, pVal) in zip(keys(calibrated_params), values(calibrated_params)) # copied from TCRunner.jl
        if pName != "case_name" # don't need to overwirte the case_name since the namelist was generated for them
            @info pName, pVal
            local param_subdict # so that it is accessible outside the try block
            try
                param_subdict = get_subdict_with_key(namelist, pName) # get the subdict in namelist
            catch e
                @warn(e, "looking for alternate instruction in alt_param_placements...")
                if pName in keys(alt_param_placements)
                    @info(
                        "using known alternate paramter placement in namelist subdict " *
                        string(alt_param_placements[pName]) *
                        " for parameter " *
                        pName
                    )
                else
                    error(
                        "key $pName not found in namelist and no alternate placement specified in alt_param_placements",
                    )
                end
                param_subdict = CalibrateEDMF.HelperFuncs.get_last_nested_dict(namelist, alt_param_placements[pName]) # need functionality to go deeper I guess...
            end
            param_subdict[pName] = pVal # update it with our calibrated output param
        end
    end

    # update_namelist!(namelist, setup) # update_namelist() from HelperFuncs.jl which uses change_entry! and namelist_subdict_by_key(), i think we can't use bc you need to specify (subdict, key, newval) like in the config.jl files, maybe there's a case where the key isn't in the namelist already? idk...
    for (pName, pVal) in zip(keys(setup), values(setup)) # copied from TCRunner.jl
        if pName != "case_name" # don't need to overwirte the case_name since the namelist was generated for them
            @info pName, pVal
            local param_subdict # so that it is accessible outside the try block
            try
                param_subdict = get_subdict_with_key(namelist, pName) # get the subdict in namelist
            catch e
                @warn(e, "looking for alternate instruction in alt_param_placements...")
                if pName in keys(alt_param_placements)
                    @info(
                        "using known alternate paramter placement in namelist subdict " *
                        string(alt_param_placements[pName]) *
                        " for parameter " *
                        pName
                    )
                else
                    error(
                        "key $pName not found in namelist and no alternate placement specified in alt_param_placements",
                    )
                end
                param_subdict = CalibrateEDMF.HelperFuncs.get_last_nested_dict(namelist, alt_param_placements[pName]) # need functionality to go deeper I guess...
            end
            param_subdict[pName] = pVal # update it with our calibrated output param
        end
    end

    # set the output directory (output_dir above is the parent one but we can set levels for each run still I believe..., outherwise the Output.simname directory goes directly there...)
    # trying to copy format from SLURM_submit_SOCRATES.jl in TrainTau.jl
    sorted_setup = sort(collect(setup), by = x -> x[1]) # sort by key
    descriptor = rstrip(
        join([
            string(key) * "-" * string(val) * "__" for
            (key, val) in filter(p -> p[1] in name_heirarchy[0], sorted_setup)
        ]),
        '_',
    )
    name = rstrip(
        join([
            string(key) * "-" * string(round(log10(val), sigdigits = 4)) * "__" for
            (key, val) in filter(p -> p[1] in name_heirarchy[1], sorted_setup)
        ]),
        '_',
    ) # we don't have a dict w/ user_args, user_aux so this should be simpler too i think... copy same code from descriptor
    # name      is htere a way to put them in dir_array from here? we don't have the coordinates though... and if theyre merged we don't even know which parameters_list it's from...
    # @info(descriptor)
    # @info(name)
    setup_outpath = joinpath(output_dir, name, descriptor) # for now, use the TrainTau.jl nested output structure, but maybe we can streamline this later to just one directory and read from that...
    namelist["output"]["output_root"] = setup_outpath
    append!(namelists, [namelist])
end

# run the model
run_TCs(namelists; tc_output_dir = nothing) # set no output dir and use the ones from the namelists


# iterate over each paramter dict, create out_data, etc...
for (i, parameters) in enumerate(parameters_list)
    out_data_save_path = joinpath(parameters_output_dirs_list[i], "grid_search_log.out") # match format from TrainTau.jl
    out_data_save_path_pkl = joinpath(parameters_output_dirs_list[i], "grid_search_log.out.pkl")
    out_data = out_datas[i]

    dict_to_named_tuple(d) = (; (Symbol(k) => v for (k, v) in d)...)
    out_data_pkl = deepcopy(out_data) # deepcopy so we can modify it without modifying the original
    out_data_pkl["tau"] = dict_to_named_tuple(out_data_pkl["tau"])
    out_data_pkl = dict_to_named_tuple(out_data_pkl)

    @info("JLD2 saving native output as NamedTuple to: ", out_data_save_path)
    JLD2.save_object(out_data_save_path, out_data)
    @info("Pickle saving native output as Dictionary to: ", out_data_save_path_pkl)
    pickle = pyimport("pickle") # pickle and serializaiton didn't work so use this
    f = open(out_data_save_path_pkl, "w+") # could be do block i guess, then don't need close
    pickle.dump(PyObject(out_data), f)
    close(f)
end
# end

# can we just make one long name, read that in, and then split it up and label coordinates and then merge? rather than all this fanfare?
# would need to rename case_name to case and τ_{sub_dep,cond_evap} to tau_{sub_dep,cond_evap} to match our old format...
