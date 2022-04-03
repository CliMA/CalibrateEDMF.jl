#=
This is an integration test of a restarted calibration process.
=#

# Import modules to all processes
using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere begin
    using ArgParse
    using Glob
    import Random
    Random.seed!(1234)
    using CalibrateEDMF
    using CalibrateEDMF.DistributionUtils
    using CalibrateEDMF.Pipeline
    using CalibrateEDMF.HelperFuncs
    # Import EKP modules
    using EnsembleKalmanProcesses
    using EnsembleKalmanProcesses.ParameterDistributions
end
using JLD2
using Test

s = ArgParseSettings()
@add_arg_table s begin
    "--config"
    help = "Inverse problem config file path"
    arg_type = String
end
parsed_args = parse_args(ARGS, s)
config_rel_filepath = parsed_args["config"]
config_basename = first(split(config_rel_filepath, "."))

folder = joinpath(@__DIR__, "output", string(config_basename, "_julia_parallel"))
@assert isdir(folder)
outdir_path = open(f -> read(f, String), string(folder, config_basename, ".txt"))

include(joinpath(outdir_path, "config.jl"))
ekobjs = glob(joinpath(relpath(outdir_path), "ekobj_iter_*.jld2"))
iters = [parse(Int64, first(split(split(split(ekobj, "/")[end], "_")[end], "."))) for ekobj in ekobjs]
last_iteration = maximum(iters)

priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))
ekobj = load(ekobj_path(outdir_path, last_iteration))["ekp"]
config = get_config()
# continue calibration process for 5 iterations
config["process"]["N_iter"] = last_iteration + 5
restart_calibration(ekobj, priors, last_iteration, config, outdir_path)

# Dispatch SCM eval functions to workers
@everywhere begin
    scm_eval_train(version) = versioned_model_eval(version, $outdir_path, "train", $config)
    scm_eval_validation(version) = versioned_model_eval(version, $outdir_path, "validation", $config)
end
# Calibration process
N_iter = config["process"]["N_iter"]
@info "Continue EK updates for $(N_iter-last_iteration) iterations"
for iteration in (last_iteration + 1):N_iter
    @time begin
        @info "   iter = $iteration, ($(iteration - last_iteration) since restart)"
        versions = readlines(joinpath(outdir_path, "versions_$(iteration).txt"))
        ekp = load(ekobj_path(outdir_path, iteration))["ekp"]
        pmap(scm_eval_train, versions)
        pmap(scm_eval_validation, versions)
        ek_update(ekp, priors, iteration, config, versions, outdir_path)
    end
end

import NCDatasets
const NC = NCDatasets

nt = NC.Dataset(joinpath(outdir_path, "Diagnostics.nc"), "r") do ds
    (;
        mse_full_mean = Array(ds.group["metrics"]["mse_full_mean"]),
        mse_full_nn_mean = Array(ds.group["metrics"]["mse_full_nn_mean"]),
        mse_full_var = Array(ds.group["metrics"]["mse_full_var"]),
        val_mse_full_mean = Array(ds.group["metrics"]["val_mse_full_mean"]),
        val_mse_full_nn_mean = Array(ds.group["metrics"]["val_mse_full_nn_mean"]),
        val_mse_full_var = Array(ds.group["metrics"]["val_mse_full_var"]),
        phi_mean = Array(ds.group["ensemble_diags"]["phi_mean"]),
    )
end

mse_full_mean = nt.mse_full_mean[1:(end - 1)]
mse_full_nn_mean = nt.mse_full_nn_mean[1:(end - 1)]
mse_full_var = nt.mse_full_var[1:(end - 1)]
val_mse_full_mean = nt.val_mse_full_mean[1:(end - 1)]
val_mse_full_nn_mean = nt.val_mse_full_nn_mean[1:(end - 1)]
val_mse_full_var = nt.val_mse_full_var[1:(end - 1)]
phi_mean = nt.phi_mean

import Plots

# https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846
ENV["GKSwstype"] = "nul"

iters = 1:N_iter

@show iters
@show mse_full_mean
@show mse_full_nn_mean
@show mse_full_var
@show phi_mean

mse_full_nn_mean_std = mse_full_nn_mean .+ sqrt.(mse_full_var)
val_mse_full_nn_mean_std = val_mse_full_nn_mean .+ sqrt.(val_mse_full_var)

p1 = Plots.plot(iters, mse_full_nn_mean; label = "mean")
Plots.plot!(iters, mse_full_nn_mean_std; label = "")
Plots.plot!(iters, mse_full_nn_mean; fillrange = mse_full_nn_mean_std, fillalpha = 0.35, c = 1, label = "")
Plots.xlabel!("Iteration")
Plots.ylabel!("Train MSE (full)")
Plots.plot!(; left_margin = 40 * Plots.PlotMeasures.px)

p2 = Plots.plot(iters, val_mse_full_nn_mean; label = "mean")
Plots.plot!(iters, val_mse_full_nn_mean_std; label = "")
Plots.plot!(iters, val_mse_full_nn_mean; fillrange = val_mse_full_nn_mean_std, fillalpha = 0.35, c = 1, label = "")
Plots.xlabel!("Iteration")
Plots.ylabel!("Validation MSE (full)")
Plots.plot!(; left_margin = 40 * Plots.PlotMeasures.px)

Plots.plot(p1, p2; layout = (1, 2))
folder = joinpath(@__DIR__, "output", string(config_basename, "_julia_parallel"))
mkpath(folder)
Plots.savefig(joinpath(folder, string(config_basename, "_mse_continued.png")))

using Test
@testset "Julia Parallel Calibrate" begin
    # TODO: add better regression test (random seed)
    @test mse_full_mean[end] < mse_full_mean[1]
    @test mse_full_var[end] < mse_full_var[1]
end
