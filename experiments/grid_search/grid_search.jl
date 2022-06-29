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

    """
        get_parameter_pairs(namelist, p1, p2, v1, v2)

    Get dictionary of parameter name-value pairs given parameters `p1` and `p2` with values
    `v1` and `v2`, respectively. If either p1 or p2 are components of a vector parameter,
    the returned dictionary also contains name-value pairs for all components of that
    vector parameter. The other components are defined in the `namelist`.
    """
    function get_parameter_pairs(namelist::Dict, p1::S, p2::S, v1::Number, v2::Number) where {S <: AbstractString}
        turbconv_params = namelist["turbulence"]["EDMF_PrognosticTKE"]  # TODO: Generalize for vector parameters from other parts of the namelist

        # If any parameters are vectors, fetch relevant defaults from namelist
        vec_param_names = getfield.(filter(!isnothing, match.(r".*(?=_{\d+})", [p1, p2])), :match)
        namelist_vec_params = filter(p -> p.first ∈ vec_param_names, turbconv_params)
        # expand vector parameters to index form
        flatten_vector_parameter(p) = "$(p.first)_{" .* string.(1:length(p.second)) .* "}" .=> p.second
        params = Dict((map(flatten_vector_parameter, collect(namelist_vec_params))...)...)
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
        "Parameter param_map, see [`ParameterMap`](@ref) for details"
        param_map::ParameterMap
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
        s.param_map,
    )
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
        param_map::ParameterMap,
    ) where {S <: AbstractString}
        # Create path to store forward model output
        case_dir = joinpath(output_root, "$param1.$param2/$(value1)_$(value2)/$case.$case_id")  # e.g. output/220101_abc/param1.param2/0.1_0.2/Bomex.1

        # If the output simulation directory already exists, don't run a simulation for this configuration.
        output_dir = joinpath(case_dir, "Output.$case.$ens_i")
        if isdir(output_dir)
            return nothing
        end

        # Otherwise, create the case directory and run a simulation
        mkpath(case_dir)

        # Get namelist for case
        namelist = NameList.default_namelist(case; write = false, set_seed = false)

        params = get_parameter_pairs(namelist, param1, param2, value1, value2)

        # Run forward model
        run_SCM_handler(
            case,
            case_dir;
            u = collect(Float64, values(params)),
            u_names = collect(String, keys(params)),
            param_map = param_map,
            namelist = namelist,
            namelist_args = namelist_args,
            uuid = "$ens_i",
            les = get(namelist["meta"], "lesfile", nothing),
        )
    end
end  # end @everywhere

function grid_search(config::Dict, config_path::String)
    root = get_entry(config["grid_search"], "output_root_dir", pwd())
    now = Dates.format(Dates.now(), "YYmmdd")
    suffix = Random.randstring(3) # ensure output folder is unique
    out_dir = joinpath(root, "output", "$(now)_$(suffix)")
    grid_search(config, config_path, out_dir)
end
function grid_search(config::Dict, config_path::String, out_dir::String)
    sim_type = get_entry(config["grid_search"], "sim_type", "reference")
    @assert sim_type in ("reference", "validation")

    # Make output folder
    mkpath(out_dir)
    if config_path != joinpath(out_dir, "config.jl")
        cp(config_path, joinpath(out_dir, "config.jl"); force = true)
    end

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
    param_map = get_entry(get(config, "prior", Dict()), "param_map", HelperFuncs.do_nothing_param_map())  # do-nothing param map by default

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
                    Ref(param_map),
                ),
            )
        end
    end
    pmap(run_sims, sim_configs; on_error = e -> @warn "Worker failure" exception = (e, catch_backtrace()))  #, on_error = e -> NaN)
end


function parse_commandline_gs()
    s = ArgParseSettings(; description = "Run config input")

    @add_arg_table s begin
        "--config"
        help = "config file"
        arg_type = String
        default = "config.jl"
        "--outdir"
        help = "output directory"
        arg_type = String
        default = ""
        "--mode"
        help = "grid search mode (new or restart)"
        arg_type = String
        default = "new"
    end

    return ArgParse.parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline_gs()

    sim_mode = args["mode"]
    if sim_mode == "new"
        @info "Starting new grid search"
        config_path = args["config"]
        include(config_path)
        config = get_config()
        grid_search(config, config_path)
    elseif sim_mode == "restart"
        @info "Restarting / continuing existing grid search"
        outdir = args["outdir"]
        config_path = joinpath(outdir, "config.jl")
        include(config_path)
        config = get_config()
        grid_search(config, config_path, outdir)
    else
        throw(ArgumentError("Invalid mode: $sim_mode"))
    end
end
