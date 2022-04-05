using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(dirname(dirname(@__DIR__)))
using ArgParse

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

    # parameter dictionary can consist of both scalar and vector parameters.
    # flatten to only scalar parameters s.t. a vector parameter 'vec' becomes 'vec_{1}', 'vec_{2}', etc.
    function flatten_vector_parameters(param_dict::Dict)
        out_dict = Dict{String, Number}()
        for (param, value) in param_dict
            if value isa AbstractArray
                for j in 1:length(value)
                    out_dict["$(param)_{$j}"] = value[j]
                end
            else
                out_dict[param] = value
            end
        end
        return out_dict
    end

    function prepare_input_parameters(namelist::Dict, p1::S, p2::S, v1::Number, v2::Number) where {S <: String}
        turbconv_params = namelist["turbulence"]["EDMF_PrognosticTKE"]

        # If any parameters are vectors, fetch relevant defaults from namelist
        param_pair = [p1, p2]
        vec_params = param_pair[occursin.(r"(_{\d+})", param_pair)]
        vec_param_names = unique(first.(split.(vec_params, r"(_{\d+})")))
        params = flatten_vector_parameters(Dict(k => v for (k, v) in turbconv_params if k ∈ vec_param_names))
        # update params with custom parameters
        params[p1] = v1
        params[p2] = v2
        return params
    end

    struct SimConfig
        "Name of first parameter"
        parameter1::String
        "Name of second parameter"
        parameter2::String
        "Value of first parameter"
        value1::Number
        "Value of second parameter"
        value2::Number
        "Ensemble index"
        ens_i::Integer
        "Case name"
        case_name::String
        "Case name index"
        case_id::Integer
        "Simulation output root directory"
        output_root::String
        "Additional namelist arguments"
        namelist_args::Union{AbstractVector, Nothing}
    end

    """
        run_sims(t) = run_sims(t...)
        run_sims(param_values::Tuple{Number, Number}, case::String, ens_ind::Integer, nt::NamedTuple)

    Run one forward simulation of the SCM given a case and pair of parameter values to be modified.

    Parameters:
    - param_values  :: Tuple of two values that are to be modified when running the model
    - case          :: Name of the simulation case to be run (e.g. Bomex)
    - ens_ind       :: Integer value for which ensemble member is to be run 
    - nt            :: Named tuple that contains information that is constant across all simulations. 
        Has the entires `output_dir`, `param_pair`, `namelist_args`, denoting the simulation output 
        directory, names of parameters corresponding to `param_values`, and additional namelist 
        arguments to be modified with respect to the default namelist, respectively.
    """
    # run_sims(t) = run_sims(t...)
    run_sims(s::SimConfig) = run_sims(
        s.parameter1,
        s.parameter2,
        s.value1,
        s.value2,
        s.ens_i,
        s.case_name,
        s.case_id,
        s.output_root,
        s.namelist_args,
    )
    # function run_sims(param_values::Tuple{Number, Number}, case::String, ens_ind::Integer, nt::NamedTuple)
    function run_sims(
        param1::S,
        param2::S,
        value1::Number,
        value2::Number,
        ens_i::Integer,
        case::S,
        case_id::Integer,
        output_root::S,
        namelist_args::Union{AbstractVector, Nothing},
    ) where {S <: String, T <: Tuple}
        # Create path to store forward model output
        case_dir = joinpath(output_root, "$param1.$param2/$(value1)_$(value2)/$case.$case_id")  # e.g. output/220101_abc/param1.param2/0.1_0.2/Bomex.1
        mkpath(case_dir)

        # Get namelist for case
        # namelist = NameList.default_namelist(case, write=false, set_seed=false)
        namelist = NameList.default_namelist(case, write = false) # until the next release where set_seed PR will be included in TC.jl

        params = prepare_input_parameters(namelist, param1, param2, value1, value2)

        # Run forward model
        run_SCM_handler(
            case,
            case_dir;
            u = collect(Float64, values(params)),
            u_names = collect(String, keys(params)),
            namelist = namelist,
            namelist_args = namelist_args,
            uuid = "$ens_i",
            les = get(namelist["meta"], "lesfile", nothing),
        )
    end
end  # end @everywhere

function grid_search(config::Dict, config_path::String, sim_type::String)
    root = pwd()
    now = Dates.format(Dates.now(), "YYmmdd")
    suffix = Random.randstring(3) # ensure output folder is unique
    out_dir = joinpath(root, "output", "$(now)_$(suffix)")
    grid_search(config, config_path, sim_type, out_dir)
end
function grid_search(config::Dict, config_path::String, sim_type::String, out_dir::String)
    @assert sim_type in ("reference", "validation")

    # Make output folder
    mkpath(out_dir)
    cp(config_path, joinpath(out_dir, "config.jl"), force = true)

    # get config entries
    parameters = config["grid_search"]["parameters"]
    n_ens = config["grid_search"]["ensemble_size"]

    # Get cases and count repeat instances of same case names
    cases = config[sim_type]["case_name"]
    case_name_id = Tuple[]
    for (i, case) in enumerate(cases)
        case_id = length(cases[1:i][(cases .== case)[1:i]])
        push!(case_name_id, (case, case_id))
    end
    namelist_args = get_entry(config["scm"], "namelist_args", nothing)

    # Construct simulation configs
    param_names = collect(keys(parameters))
    param_pairs = combinations(param_names, 2)
    sim_configs = SimConfig[]
    for (param1, param2) in param_pairs
        for (case, case_id) in case_name_id
            config_product = Iterators.product(parameters[param1], parameters[param2], 1:n_ens)
            append!(
                sim_configs,
                SimConfig.(
                    param1,
                    param2,
                    getfield.(config_product, 1),  # param1 value
                    getfield.(config_product, 2),  # param2 value
                    getfield.(config_product, 3),  # ensemble id
                    case,
                    case_id,
                    out_dir,
                    Ref(namelist_args),
                ),
            )
        end
    end
    pmap(run_sims, sim_configs)
end


function parse_commandline_gs()
    s = ArgParseSettings(; description = "Run config input")

    @add_arg_table s begin
        "--config"
        help = "config file"
        arg_type = String
        default = "config.jl"
        "--sim_type"
        help = "Type of simulations to consider (`reference` or `validation`)"
        arg_type = String
        default = "reference"
    end

    return ArgParse.parse_args(s)  # parse_args(ARGS, s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline_gs()
    config_path = args["config"]
    sim_type = args["sim_type"]
    include(config_path)
    config = get_config()
    grid_search(config, config_path, sim_type)
end
