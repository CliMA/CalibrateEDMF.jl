using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(dirname(@__DIR__))
@everywhere using CalibrateEDMF.TurbulenceConvectionUtils
@everywhere using CalibrateEDMF.HelperFuncs
@everywhere import CalibrateEDMF.ReferenceModels: NameList
@everywhere include("DiagnosticsTools.jl")
@everywhere using Random
using ArgParse
using Dates

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
- metric           :: mse metric to find the minimum of {"mse_full", "val_mse_full"}.
- n_ens            :: Number of ensemble to run per case
"""

function run_TC_optimal(
    results_dir::String,
    tc_output_dir::String,
    config::Dict,
    run_cases::Dict;
    method::String = "best_nn_particle_mean",
    metric::String = "mse_full",
    n_ens::Int = 1,
)
    cases = run_cases["case_name"]
    # get namelist_args
    global_namelist_args = get_entry(config["scm"], "namelist_args", nothing)
    case_namelist_args = expand_dict_entry(run_cases, "namelist_args", length(cases))
    namelist_args = merge_namelist_args.(Ref(global_namelist_args), case_namelist_args)

    param_map = get_entry(config["prior"], "param_map", HelperFuncs.do_nothing_param_map())  # do-nothing param map by default
    u_names, u = optimal_parameters(joinpath(results_dir, "Diagnostics.nc"); method = method, metric = metric)

    @everywhere run_single_SCM(t::Tuple) = run_single_SCM(t...)
    @everywhere run_single_SCM(case_nt::NamedTuple, ens_ind::Integer) = begin
        case = case_nt.case
        @info "Running $(case) ($(case_nt.case_id)). Ensemble member $ens_ind."
        # Get namelist for case
        namelist = NameList.default_namelist(case, write = false, set_seed = false)
        # Set optional namelist args
        update_namelist!(namelist, $namelist_args)
        # Run TC.jl
        run_SCM_handler(
            case,
            $tc_output_dir;
            u = $u,
            u_names = $u_names,
            param_map = $param_map,
            namelist = namelist,
            uuid = "$(case_nt.case_id)_$(ens_ind)",
            les = case_nt.les_path,
        )
    end

    case_nt = NamedTuple[]
    for (i, case) in enumerate(cases)
        case_id = length(cases[1:i][(cases .== case)[1:i]])
        les_path = (case == "LES_driven_SCM") ? get_stats_path(run_cases["y_dir"][i]) : nothing
        push!(case_nt, (; case, case_id, les_path))
    end

    case_ens = Iterators.product(case_nt, 1:n_ens)
    @info "Preparing to run $(length(cases)) cases × $n_ens ensembles = $(length(cases)*n_ens) total forward model evaluations."
    pmap(run_single_SCM, case_ens)
    @info "Finished. Current time: $(Dates.now())"
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

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    # get config file
    results_dir = abspath(args["results_dir"])
    include(joinpath(results_dir, "config.jl"))
    config = get_config()
    tc_output_dir = if !isnothing(args["tc_output_dir"])
        abspath(args["tc_output_dir"])
    else
        joinpath(results_dir, "fwd_map")
    end

    @info "Running TCRunner.jl. Current time: $(Dates.now())"
    @info "Running TC for $(results_dir)"
    @info "Storing output in $(tc_output_dir)"

    # get case set
    run_set = args["run_set"]
    @info "Running TC on $run_set set"
    run_cases = if run_set in ("reference", "validation")
        config[run_set]
    elseif run_set == "test"
        run_set_config = args["run_set_config"]
        isnothing(run_set_config) && throw(ArgumentError("--run_set_config must be specified if --run_set is test"))
        @info "Fetching $run_set set from $run_set_config"
        include(run_set_config)
        get_reference_config(ScmTest())
    else
        throw(ArgumentError("--run_set must be one of {reference, validation, test}"))
    end

    @info "Using method $(args["method"]) and metric $(args["metric"]) to compute optimal parameters"

    run_TC_optimal(
        results_dir,
        tc_output_dir,
        config,
        run_cases;
        method = args["method"],
        metric = args["metric"],
        n_ens = args["n_ens"],
    )
end
