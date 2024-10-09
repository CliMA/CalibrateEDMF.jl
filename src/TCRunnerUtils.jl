module TCRunnerUtils # I think we can't have a module that works with distributed, see  https://discourse.julialang.org/t/error-running-distributed-code-inside-of-a-module/54283/6

# maybe bring everything to a separate module except runTCs()
export create_namelists_from_calibration_output,
    get_calibrated_parameters,
    # run_TCs,
    recursive_merge,
    run_requested_parameters,
    get_subdict_with_key,
    run_one_SCM


using ..TurbulenceConvectionUtils # what do we use this for? it's redundant no? we at least get create_parameter_vectors()but that's exported
using ..HelperFuncs
using ..ReferenceModels # for data_directory
import ..ReferenceModels: NameList
# include("DiagnosticsTools.jl")
using Random
using ArgParse
using Dates

using Dates
using ArgParse

using TurbulenceConvection # also like TurbulenceConvectionUtils.jl to give us main1d
tc = pkgdir(TurbulenceConvection)
# include(joinpath(tc, "driver", "main.jl")) # define the main.jl function... (overly broad, no? why doesn't TC just export this lol...) this i think redundatnly defines the things in main.jl... how to get around that?
main1d = TurbulenceConvectionUtils.main1d # do this bc we don't wanna redundantly import main.jl bc it's included in in TurbulenceConvectionUtils.jl

# these don't seem to be working inside the @everywhere block? idk why... race condition bug?
# using CalibrateEDMF
# pkg_dir = pkgdir(CalibrateEDMF)
pkg_dir = joinpath(@__DIR__, "..")
include(joinpath(pkg_dir, "tools", "DiagnosticsTools.jl")) # for final_paramters/optimal_parameters

"""
For this file, I want to enable running any set of input cases (not just with the output from calibration)

We would like to enable
- getting a namelist from a calibration output
- running a set of case with a set of namelists
- outputing all the outputs in one place using a distributed framework

This would in principle allow us to do the sort of thing we did in TrainTau.jl, running grids of inputs, but with the additional functionality of using calibrated parameters and much easier slurm integration.
This is also good because we can take parameters from multiple different calibrations and combine them into one namelist and use it here

currently even merging dicts normally doesnt work bc stuff like config["turbulence"]["EDMF_PrognosticTKE"]["nn_ent_params"] don't match between runs

"""

@info final_parameters
# typeof(final_parameters)

"""
    run_TC_optimal(results_dir, tc_output_dir, config, run_cases; 
                  [method = "best_nn_particle_mean", metric = "mse_full", n_ens = 1])

Given path to the results directory of completed calibration run and associated config, run TC
using optimal parameters on a case set and save the resulting TC stats files.

# Arguments
- results_dir      :: directory containing CEDMF `Diagnostics.nc` file.
- tc_output_dir    :: directory to store TC output
- config           :: config dictionary

# Keyword arguments
- method           :: method for computing optimal parameters. Use parameters of:
    "best_particle" - particle with lowest mse in training (`metric` = "mse_full") or validation (`metric` = "mse_full_val") set.
    "best_nn_particle_mean" - particle nearest to ensemble mean for the iteration with lowest mse.
    "final_mean" - ensemble mean at final iteration.
- metric           :: mse metric to find the minimum of {"mse_full", "val_mse_full"}.
- n_ens            :: Number of ensemble to run per case
"""

"""
This allows us to gather the namelist from one calibration output
However we still need to figure out how to merge namelists from different parameter calibrations.
"""
function create_namelists_from_calibration_output(
    results_dir::String;
    uuid::String = "01", # from TurbulenceConvectionUtils.jl run_SCM_handler()
    out_dir::String = "./", # default to here since that's what it got set to in config.
    method::String = "final_mean", # the method for condensing the calibration ensemble into actual parameters
)
    # get config file
    @info("Retrieving config file info from $(results_dir)")
    results_dir = abspath(results_dir)
    include(joinpath(results_dir, "config.jl")) # load config file,  sometimes this seems to give world age errors from redefining get_config(), sometimes it doesnt... idk...
    config = get_config() # get the config from the config spec
    case_names = config["reference"]["case_name"]

    @info("Assembling config files namelist specifications")
    # get global namelist_args
    global_namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    case_namelist_args = expand_dict_entry(config["reference"], "namelist_args", length(case_names)) # get the namelist from the reference cases we trained on... (not the validation)
    # Assemble namelist_args per case
    namelist_args = merge_namelist_args.(Ref(global_namelist_args), case_namelist_args)

    # Get namelist for case
    @info("Generating default_namelists for cases $(case_names)")
    namelists = NameList.default_namelist.(case_names, write = false, set_seed = false)
    # Set optional namelist args
    @info("Overwriting/Updating default_namelists with config specifications")
    update_namelist!.(namelists, namelist_args)


    ##############################
    # From TurbulenceConvectionUtils.jl run_SCM_handler()
    ##############################
    @info("Updating parameters with calibrated values ")
    # we could use get_calibrated_parameters() here, but that relies on creating another namelist, so we just copy that code here
    for (case_name, namelist) in zip(case_names, namelists)
        namelist["meta"]["uuid"] = uuid
        # skip setting output_dir here like in scm_handler since this just supposed to be namelist that you can pass around arbitrarily, default is "./"
        namelist["output"]["output_root"] = out_dir

        param_map = get_entry(config["prior"], "param_map", HelperFuncs.do_nothing_param_map())  # do-nothing param map by default (see https://clima.github.io/CalibrateEDMF.jl/dev/API/HelperFuncs/#CalibrateEDMF.HelperFuncs.ParameterMap)
        if method == "final_mean"
            u_names, u = final_parameters(joinpath(results_dir, "Diagnostics.nc")) # from tools/DiagnosticsTools.jl
        else
            u_names, u = optimal_parameters(joinpath(results_dir, "Diagnostics.nc"); method = method, metric = metric)
        end

        u_names, u = create_parameter_vectors(u_names, u, param_map, namelist)
        @info(Dict(u_names .=> u))

        # update learnable parameter values
        @assert length(u_names) == length(u)
        for (pName, pVal) in zip(u_names, u)
            param_subdict = namelist_subdict_by_key(namelist, pName) # get the subdict in namelist
            param_subdict[pName] = pVal # update it with our calibrated output param
        end

        # LES handler from TurbulenceConvectionUtils.jl run_SCM_handler()
        if case_name == "LES_driven_SCM" 
            if isnothing(les)
                error("les path or keywords required for LES_driven_SCM case!")
            elseif isa(les, NamedTuple)
                les = get_stats_path(
                    get_cfsite_les_dir(
                        les.cfsite_number;
                        forcing_model = les.forcing_model,
                        month = les.month,
                        experiment = les.experiment,
                    ),
                )
            end
            namelist["meta"]["lesfile"] = les
        end
    end

    return namelists

end




"""
Collect the values of the calibrated parameters (not in namelist format, just a simple dict)
Ideally we wouldn't also create the namelist, but it seems necessary for using param_map and create_parameter_vectors etc...
"""
function get_calibrated_parameters(
    results_dir::String; # the directory of the calibration
    method::String = "final_mean", # the method for condensing the calibration ensemble into actual parameters
)

    # get config file
    @info("Retrieving config file info from $(results_dir)")
    results_dir = abspath(results_dir)
    include(joinpath(results_dir, "config.jl")) # load config file,  sometimes this seems to give world age errors from redefining get_config(), sometimes it doesnt... idk...
    # config = get_config() # get the config from the config spec (causes world age problems)
    config = Base.invokelatest(get_config) # seems to help w/ world age problem...
    case_names = config["reference"]["case_name"]

    @info("Assembling config files namelist specifications")
    # get global namelist_args
    global_namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    case_namelist_args = expand_dict_entry(config["reference"], "namelist_args", length(case_names)) # get the namelist from the reference cases we trained on... (not the validation)
    # Assemble namelist_args per case
    namelist_args = merge_namelist_args.(Ref(global_namelist_args), case_namelist_args)

    # Get namelist for case
    @info("Generating default_namelists for cases $(case_names)")
    namelists = NameList.default_namelist.(case_names, write = false, set_seed = false)
    # Set optional namelist args
    @info("Overwriting/Updating default_namelists with config specifications")
    update_namelist!.(namelists, namelist_args)

    @info final_parameters

    ##############################
    # From TurbulenceConvectionUtils.jl run_SCM_handler()
    ##############################
    @info("Updating parameters with calibrated values ")
    calibrated_param_sets = []
    for (case_name, namelist) in zip(case_names, namelists) # do we need to do this? the calibration returns one joint calibrated output... (but I guess it's possible different if there are param_maps going into vectors? )
        # namelist["meta"]["uuid"] = uuid
        # skip setting output_dir here like in scm_handler since this just supposed to be namelist that you can pass around arbitrarily, default is "./"
        # namelist["output"]["output_root"] = out_dir

        param_map = get_entry(config["prior"], "param_map", HelperFuncs.do_nothing_param_map())  # do-nothing param map by default (see https://clima.github.io/CalibrateEDMF.jl/dev/API/HelperFuncs/#CalibrateEDMF.HelperFuncs.ParameterMap)
        if method == "final_mean"
            u_names, u = final_parameters(joinpath(results_dir, "Diagnostics.nc"))
        else
            u_names, u = optimal_parameters(joinpath(results_dir, "Diagnostics.nc"); method = method, metric = metric)
        end

        u_names, u = create_parameter_vectors(u_names, u, param_map, namelist)
        calibrated_param_set = Dict(u_names .=> u)
        @info(calibrated_param_set)
        append!( calibrated_param_sets, [calibrated_param_set] )
    end

    return calibrated_param_sets
end


"""
Recursively merge dictionaries so we can merge namelists...
Need to add some sort of conflict handler though (maybe it already fails on conflicts?
 -- update it does fail on conflicts, but it doesn't tell you where the conflicts are)
"""
function recursive_merge(x::AbstractDict...; resolve_conflict=false)
    merger = (x...) -> recursive_merge(x...;resolve_conflict=resolve_conflict)
    return mergewith(merger, x...;) # https://discourse.julialang.org/t/multi-layer-dict-merge/27261/2 
end

"""
Handles the leaves of the recursive merge, to see if there's any conflicts. defaults to taking the value from the last dict
"""
function recursive_merge(x...; resolve_conflict=true)
    if resolve_conflict || allequal([x...]) # if we want to resolve conflicts, or if all the leaves are equal and there is no conflict
        x[end]
    elseif resolve_conflict
        x[end] # https://discourse.julialang.org/t/multi-layer-dict-merge/27261/6, if we want to forgo conflict resolution, or if our keys don't conflict, take the last dict
    else # if we don't want to resolve conflicts, and there are conflicts, throw an error
        error("Cannot resolve conflicting leaves with values: ", x...)
    end
end


function run_one_SCM( # this needs to get defined everywhere with @everywhere... might move it to TC module...
    namelist::Dict;
    tc_output_dir::Union{Nothing,String}=nothing,
    )

    if isnothing(tc_output_dir)
        @info("tc_output_dir not specified; run using its default namelist[\"output\"][\"output_root\"]: " * namelist["output"]["output_root"])
    else
        @info("Overwriting output root in namelist[\"output\"][\"output_root\"] with specified tc_output_dir: " * tc_output_dir)
        namelist["output"]["output_root"] = tc_output_dir # by default this is nothing, and the default in each namelist should prevail (usually is "./")
    end

    model_error = false
    case_name = namelist["meta"]["simname"] # i think this what we want for the fullname
    uuid = namelist["meta"]["uuid"]
    out_dir = namelist["output"]["output_root"]

    # run TurbulenceConvection.jl with modified parameters
    logger = Logging.ConsoleLogger(stderr, Logging.Warn)
    _, _, ret_code = Logging.with_logger(logger) do
        @info("launching")
        try
            main1d(namelist; time_run = false)  # from run_SCM_handler() in TurbulenceConvectionUtils.jl
        catch e
            @warn e
            ret_code = :failure
            return nothing,nothing,ret_code # return ret_code, skip others...
        end
    end
    if ret_code ≠ :success
        model_error = true
        message = ["TurbulenceConvection.jl simulation $out_dir, $uuid failed: \n"] # can't rly print params cause we didn't store that here lol... oops. maybe we need a function to take parameters and values to create a meshgrid of and run and alert failures...
        # append!(message, ["$param_name = $param_value \n" for (param_name, param_value) in zip(u_names, u)])
        @warn join(message)
    end

    return data_directory(out_dir, case_name, uuid), model_error # data_directory is where TC.jl puts things inside the output_dir

end





"""
A function to take requested parameters, and run TC with them...
we might also need a list of case_names? to generate the base namelist which we'll insert the requested parameters into...
Also the base namelist needs stuff like t_max and stuff.... etc... now sure how to handle that... we're not pulling it from config here so idk.... maybe just a list of base_namelists? or namelist_args?
"""

function run_requested_parameters(
    requested_param_sets::Vector{Dict{Any,Any}},
    case_names::Vector{String},
    )
    return
end

"""
Given a dictionary, returns the first subdict that contains key
Just recursively looks
"""
function get_subdict_with_key(dict::Dict, key::String; queue=[])
    # @info keys(dict)
    for (k,v) in dict
        # @info(k)
        if k == key
            return dict
        elseif isa(v, Dict)
            append!(queue,[v])
        end
    end

    if length(queue) > 0
        # @info queue
        return get_subdict_with_key(popfirst!(queue), key; queue=queue)
    else
        error("No subdict with value $key found in dict")
    end
end


##

"To run from command line with e.g. 3 processors, use the format: julia -p3 TCRunner.jl --results_dir <path/to/CEDMF_output>"
function parse_commandline()

    s = ArgParseSettings()
    @add_arg_table s begin
        "--results_dir"
        help = "Path to the results directory (CEDMF output)"
        arg_type = String
        "--tc_output_dir"
        help = "Path to store TC output"
        arg_type = String
        default = nothing
        "--run_set"
        help = "Set of cases to run TC. {reference, validation, test}."
        arg_type = String
        default = "validation"
        "--run_set_config"
        help = "Path to config if test is used for --run_set."
        arg_type = String
        default = nothing
        "--method"
        help = "Method for computing optimal parameters"
        arg_type = String
        default = "best_nn_particle_mean"
        "--metric"
        help = "mse metric to find the minimum of"
        arg_type = String
        default = "mse_full"
        "--n_ens"
        help = "number of ensembles to run"
        arg_type = Int64
        default = 1
    end

    return ArgParse.parse_args(s)
end

end # module
