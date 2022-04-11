using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(dirname(dirname(@__DIR__)))
using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--config"
    help = "config file"
    arg_type = String
    default = "config.jl"
    "--out_dir"
    help = "output directory"
    arg_type = String
    default = "output"
end
parsed_args = parse_args(ARGS, s)
config_path = parsed_args["config"]
out_dir = parsed_args["out_dir"]
include(config_path)

@everywhere begin
    import Dates
    import Random
    using ArgParse
    using CalibrateEDMF
    using CalibrateEDMF.HelperFuncs
    using CalibrateEDMF.DistributionUtils
    using CalibrateEDMF.TurbulenceConvectionUtils
    using Combinatorics
    using TurbulenceConvection
    tc = pkgdir(TurbulenceConvection)
    include(joinpath(tc, "driver", "generate_namelist.jl"))

    run_sims(t) = run_sims(t...)
    function run_sims(param_values, case, j, nt)
        param_dir = joinpath(nt.output_dir, "$(join(param_values, "_"))")  # output_2020-01-01_10_30/0.1_0.1
        case_dir = joinpath(param_dir, "$case")  # output_2020-01-01_10_30/0.1_0.1/Bomex
        mkpath(case_dir)
        # namelist = NameList.default_namelist(case, set_seed=false)
        namelist = NameList.default_namelist(case) # until the next release where set_seed PR will be included in TC.jl
        run_SCM_handler(
            case,
            case_dir;
            u = collect(param_values),
            u_names = nt.param_pair,
            namelist = namelist,
            namelist_args = nt.namelist_args,
            uuid = "$j",
            les = get(namelist["meta"], "lesfile", nothing),
        )
    end
end  # end @everywhere

function grid_search(
    sim_type = "reference",
    root = pwd(),
)
    @assert sim_type in ("reference", "validation", )
    # fetch config file
    config = get_config()

    # get entries
    parameters = get_entry(config["grid_search"], "parameters", nothing)
    n_ens = get_entry(config["grid_search"], "ensemble_size", nothing)

    case_names = get_entry(config[sim_type], "case_name", nothing)
    namelist_args = get_entry(config["scm"], "namelist_args", nothing)

    # Make output folder
    now = Dates.format(Dates.now(), "YYmmdd")
    suffix = Random.randstring(3) # ensure output folder is unique
    out_dir = joinpath(root, "output", "$(now)_$(suffix)")
    mkpath(out_dir)
    cp(config_path, joinpath(out_dir, "config.jl"), force=true)

    # Loop through parameter pairs
    param_names = collect(keys(parameters))
    for param_pair in combinations(param_names, 2)
        A = parameters[param_pair[1]]
        B = parameters[param_pair[2]]
        param_values = vec(collect(Iterators.product(A, B)))
        # output dir
        param_string = join(param_pair, ".")
        output_dir = joinpath(out_dir, "$(param_string)")
        mkpath(output_dir)  # output/220101_abc/param1.param2

        nt = (;output_dir, param_pair, namelist_args)
        sim_configs = vec(collect(Iterators.product(param_values, case_names, 1:n_ens, [nt] )))
        # run simulations
        pmap(run_sims, sim_configs)
    end
end

grid_search()
