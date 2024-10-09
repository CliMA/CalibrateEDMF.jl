## Probe model failures to see regions of failure...
using Pkg
CEDMF_dir::String = "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl"
if Base.normpath(dirname(Base.active_project())) != Base.normpath(CEDMF_dir)
    Pkg.activate(CEDMF_dir)
end
using JLD2
using ProgressMeter
using StaticArrays # needed to load namelist
using StatsBase

FT = Float64

# experiment::Symbol = :Base
experiment::Symbol = :powerlaw_T_scaling_ice
# experiment::Symbol = :geometric_liq__powerlaw_T_scaling_ice
# experiment::Symbol = :linear_combination
# experiment::Symbol = :linear_combination_with_w



calibration_setup::String = "tau_autoconv_noneq"
# dt_string =  "adapt_dt__dt_min_0.5__dt_max_2.0"
# dt_string =  "adapt_dt__dt_min_2.0__dt_max_8.0"
dt_string::String = "adapt_dt__dt_min_5.0__dt_max_20.0"

adapt_dt::Bool = occursin("adapt_dt", dt_string) ? true : false
dt_min::FT = parse(FT, match(r"dt_min_(\d+\.\d+)", dt_string).captures[1])
dt_max::FT = parse(FT, match(r"dt_max_(\d+\.\d+)", dt_string).captures[1])

SOCRATES_dir::String = joinpath(CEDMF_dir, "experiments", "SOCRATES")

param_results_dir::String = joinpath(SOCRATES_dir, "global_parallel", "slurm", "param_results")

_adapt_dt::Bool = @isdefined(_adapt_dt) ? _adapt_dt : false
_dt_min::FT = @isdefined(_dt_min) ? _dt_min : FT(0)
_dt_max::FT = @isdefined(_dt_max) ? _dt_max : FT(0)
_experiment::Symbol = @isdefined(_experiment) ? _experiment : Symbol()
_calibration_setup::String = @isdefined(_calibration_setup) ? _calibration_setup : ""


# subdir_limit::Int = Int(10^10)
subdir_limit::Int = 1000
# subdir_limit::Int = 100 # Probably don't need every directory to probe failures

if true ||
   !@isdefined(results) ||
   (_experiment != experiment) ||
   (_calibration_setup != calibration_setup) ||
   (_adapt_dt != adapt_dt) ||
   (_dt_min != dt_min) ||
   (_dt_max != dt_max)



    #
    param_results_subdirs = readdir(param_results_dir; join = true) # list of directories
    # remove files keeping only directories
    param_results_subdirs = filter(x -> isdir(x), param_results_subdirs)

    param_results_subdirs = param_results_subdirs[sample(
        1:length(param_results_subdirs),
        min(subdir_limit, length(param_results_subdirs)),
        replace = false,
    )]

    results = Dict{String, Dict{String, Vector{<:Union{FT, String, Int}}}}(
        "success" =>
            Dict{String, Vector{<:Union{FT, String, Int}}}("folders" => String[], "flight_number" => Int[]),
        "failure" =>
            Dict{String, Vector{<:Union{FT, String, Int}}}("folders" => String[], "flight_number" => Int[]),
    )


    @showprogress for subdir in param_results_subdirs
        subdir_folder = basename(subdir)
        param_result_path = joinpath(subdir, "param_results.jld2")
        param_result = JLD2.load(param_result_path)["param_results"]

        timedir = param_result["namelist"]["time_stepping"]
        global _adapt_dt = timedir["adapt_dt"]
        global _dt_min = timedir["dt_min"]
        global _dt_max = timedir["dt_max"]
        global _experiment = param_result["namelist"]["user_args"][:use_supersat]
        global _calibration_setup =
            param_result["namelist"]["thermodynamics"]["moisture_model"] == "nonequilibrium" ? "tau_autoconv_noneq" :
            "pow_icenuc_autoconv_eq"

        local flight_number = param_result["namelist"]["meta"]["flight_number"]

        # println(_experiment,  , " | ", param_result["success"])

        if (_experiment == experiment) &&
           (_calibration_setup == calibration_setup) &&
           (_adapt_dt == adapt_dt) &&
           (_dt_min == dt_min) &&
           (_dt_max == dt_max)
            success_key = param_result["success"] ? "success" : "failure"
            param_result["success"] ? push!(results["success"]["folders"], subdir_folder) :
            push!(results["failure"]["folders"], subdir_folder)
            param_result["success"] ? push!(results[success_key]["flight_number"], flight_number) :
            push!(results[success_key]["flight_number"], flight_number)

            for (param, param_val) in zip(param_result["param_names"], param_result["param_vals"])
                haskey(results[success_key], param) ? append!(results[success_key][param], param_val) :
                results[success_key][param] = FT[param_val]
            end
        end
    end
end

# ============================================================================= #
# Scatter plot success and failures for pairs of parameters

using Plots; #

plot_params_sucess = collect(keys(results["success"]))
plot_param_failure = collect(keys(results["failure"]))

if length(plot_params_sucess) == 0
    @warn "No successful runs found"
    plot_params = length(plot_param_failure) > 2 ? plot_param_failure : error("Not enough parameters to plot")
elseif length(plot_param_failure) == 0
    @warn "No failed runs found"
    plot_params = length(plot_params_sucess) > 2 ? plot_params_sucess : error("Not enough parameters to plot")
else
    plot_params = intersect(plot_params_sucess, plot_param_failure)
    if length(plot_params) == 0
        @warn "No common parameters found"
    elseif length(plot_params) < 2
        error("Not enough parameters to plot")
    end
end

# Differnt maker
# markers = Dict(
#     1 => "1",
#     9 => "9"
#     10 => "10",
#     12 => "12",
#     13 => "13",
# )

# ----------------------------------------------------------------------------- #
# Note the plots break when length(plot_params) is large
# plot_params = plot_params[1:10]
# plot_params = ["powerlaw_T_scaling_ice_c_1" "powerlaw_T_scaling_ice_c_2"]
hide_params = [
    "folders" # we added this one
    "rain_sedimentation_scaling_factor"
    "surface_area"
    "q_liq_threshold"
    "snow_sedimentation_scaling_factor"
    "τ_acnv_rai"
    "ice_sedimentation_scaling_factor"
    "area_limiter_scale"
    "q_ice_threshold"
    "r_ice_snow"
    "τ_acnv_sno"
    "area_limiter_power"
]
plot_params = filter(x -> x ∉ hide_params, plot_params)

# ----------------------------------------------------------------------------- #

flight_numbers = Int[1, 9, 10, 12, 13]
num_subplots = length(plot_params)

# create n x n grid of subplots
plot_grid = Dict()
grid_layouts = Dict()
for flight_number in [flight_numbers..., "combined"]
    grid_layouts[flight_number] = Plots.@layout [Plots.grid(num_subplots, num_subplots)]
    plot_grid[flight_number] = plot(
        layout = grid_layouts[flight_number],
        size = (800 * num_subplots, 800 * num_subplots),
        left_margin = 0Plots.mm,
        right_margin = 0Plots.mm,
        top_margin = 0Plots.mm,
        bottom_margin = 0Plots.mm,
    )
end

# xscale = :log10
# yscale = :log10

xscale = :identity
yscale = :identity

@showprogress for i in 1:num_subplots
    for j in 1:num_subplots
        if j < i
            # hide axes
            plot!(subplot = (i - 1) * num_subplots + j, showaxis = false)
            continue
        end
        i_subplot = (i - 1) * num_subplots + j
        # plot on i,j
        _plot_params = (plot_params[i], plot_params[j])

        # combined
        for flight_number in [flight_numbers..., "combined"]
            if (_plot_params[1] ∈ plot_param_failure) && (_plot_params[2] ∈ plot_param_failure)
                valid =
                    flight_number == "combined" ? (1:length(results["failure"]["flight_number"])) :
                    findall(results["failure"]["flight_number"] .== flight_number)
                # additional_kwargs = flight_number == "combined" ? Dict() : Dict("series_annotations" => text.(results["failure"]["flight_number"][valid], color = "red"), "markersize" => 0, "markeralpha" => 0)
                if flight_number == "combined"
                    scatter!(
                        plot_grid[flight_number],
                        results["failure"][_plot_params[1]][valid],
                        results["failure"][_plot_params[2]][valid],
                        label = "failure",
                        color = "red",
                        markerstrokewidth = 0,
                        subplot = i_subplot,
                        left_margin = 0Plots.mm,
                        right_margin = 0Plots.mm,
                        top_margin = 0Plots.mm,
                        bottom_margin = 0Plots.mm,
                        xaxis = xscale,
                        yaxis = yscale,
                        series_annotations = text.(results["failure"]["flight_number"][valid], color = "red"),
                        markersize = 0,
                        markeralpha = 0,
                    )
                else
                    scatter!(
                        plot_grid[flight_number],
                        results["failure"][_plot_params[1]][valid],
                        results["failure"][_plot_params[2]][valid],
                        label = "failure",
                        color = "red",
                        markerstrokewidth = 0,
                        subplot = i_subplot,
                        left_margin = 0Plots.mm,
                        right_margin = 0Plots.mm,
                        top_margin = 0Plots.mm,
                        bottom_margin = 0Plots.mm,
                        xaxis = xscale,
                        yaxis = yscale,
                        markersize = 10,
                    )
                end
                # annotate!(plot_grid[flight_number], results["failure"][_plot_params[1]], results["failure"][_plot_params[2]], results["failure"]["flight_number"], annotationtextcolor="red", subplot = i_subplot, fontsize = 8)
            end
            if (_plot_params[1] ∈ plot_params_sucess) && (_plot_params[2] ∈ plot_params_sucess)
                valid =
                    flight_number == "combined" ? (1:length(results["success"]["flight_number"])) :
                    findall(results["success"]["flight_number"] .== flight_number)
                if flight_number == "combined"
                    scatter!(
                        plot_grid[flight_number],
                        results["success"][_plot_params[1]][valid],
                        results["success"][_plot_params[2]][valid],
                        label = "success",
                        color = "green",
                        markerstrokewidth = 0,
                        subplot = i_subplot,
                        left_margin = 0Plots.mm,
                        right_margin = 0Plots.mm,
                        top_margin = 0Plots.mm,
                        bottom_margin = 0Plots.mm,
                        xaxis = xscale,
                        yaxis = yscale,
                        series_annotations = text.(results["success"]["flight_number"][valid], color = "green"),
                        markersize = 0,
                        markeralpha = 0,
                    )
                else
                    scatter!(
                        plot_grid[flight_number],
                        results["success"][_plot_params[1]][valid],
                        results["success"][_plot_params[2]][valid],
                        label = "success",
                        color = "green",
                        markerstrokewidth = 0,
                        subplot = i_subplot,
                        left_margin = 0Plots.mm,
                        right_margin = 0Plots.mm,
                        top_margin = 0Plots.mm,
                        bottom_margin = 0Plots.mm,
                        xaxis = xscale,
                        yaxis = yscale,
                        markersize = 10,
                    )
                end
                # annotate!(plot_grid[flight_number], results["success"][_plot_params[1]], results["success"][_plot_params[2]], results["success"]["flight_number"], annotationtextcolor="green", subplot = i_subplot, fontsize = 8)
            end

            xlabel!(plot_params[i], subplot = i_subplot, fontsize = 10)
            ylabel!(plot_params[j], subplot = i_subplot, fontsize = 10)
        end



    end
end

# xlabel!(plot_params[1])
# ylabel!(plot_params[2])
for flight_number in [flight_numbers..., "combined"]
    plot!(
        plot_grid[flight_number],
        plot_title = "Success and Failure Scatter Plot | $experiment | $calibration_setup | $dt_string",
    )
    savefig(joinpath(SOCRATES_dir, "param_failure_scatter_$flight_number.png"))
end
