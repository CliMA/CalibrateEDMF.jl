#=
This is an integration test using Julia's pmap parallelization.

This is an example on training the TurbulenceConvection.jl implementation
of the EDMF scheme with data generated using PyCLES (an LES solver) or
TurbulenceConvection.jl (perfect model setting).

This example parallelizes forward model evaluations using Julia's pmap() function.
This parallelization strategy is efficient for small ensembles (N_ens < 20). For
calibration processes involving a larger number of ensemble members, parallelization
using HPC resources directly is advantageous.
=#

# Import modules to all processes
using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere begin
    using ArgParse
    import Random
    Random.seed!(1234)
    using CalibrateEDMF
    using CalibrateEDMF.DistributionUtils
    using CalibrateEDMF.Pipeline
    src_dir = dirname(pathof(CalibrateEDMF))
    using CalibrateEDMF.HelperFuncs
    # Import EKP modules
    using EnsembleKalmanProcesses
    using EnsembleKalmanProcesses.ParameterDistributions
end
using JLD2
using Test

config_filename = joinpath(dirname(@__DIR__), "experiments", "scm_pycles_pipeline", "config.jl")
include(config_filename)
config = get_config()

# Initialize calibration process
outdir_path = init_calibration(config; config_path = config_filename, mode = "pmap")
priors = deserialize_prior(load(joinpath(outdir_path, "prior.jld2")))

# Dispatch SCM eval functions to workers
@everywhere begin
    scm_eval_train(version) = versioned_model_eval(version, $outdir_path, "train", $config)
    scm_eval_validation(version) = versioned_model_eval(version, $outdir_path, "validation", $config)
end
# Calibration process
config["process"]["N_iter"] = 20
N_iter = config["process"]["N_iter"]
@info "Running EK updates for $N_iter iterations"
for iteration in 1:N_iter
    @time begin
        @info "   iter = $iteration"
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
folder = joinpath(@__DIR__, "output", first(split(basename(@__FILE__), ".")))
mkpath(folder)
Plots.savefig(joinpath(folder, "mse.png"))

using Test
@testset "Julia Parallel Calibrate" begin
    # TODO: add better regression test (random seed)
    @test mse_full_mean[end] < mse_full_mean[1]
    @test mse_full_var[end] < mse_full_var[1]
end
