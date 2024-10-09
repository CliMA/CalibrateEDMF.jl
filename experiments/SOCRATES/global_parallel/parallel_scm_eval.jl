"""Evaluates a set of SCM configurations for a single parameter vector."""

@everywhere begin
    using Pkg
    thisdir = @__DIR__
    @info("thisdir: $thisdir")
    CEDMF_path = abspath(joinpath(thisdir, "..", "..", ".."))
    @info("CEDMF_path: $CEDMF_path")
    Pkg.activate(CEDMF_path)
    # Pkg.activate("../../..") # @__DIR__
end

@everywhere begin
    using ArgParse
    using JLD2
    using CalibrateEDMF
    using CalibrateEDMF.Pipeline

    src_dir = dirname(pathof(CalibrateEDMF))
    cedmf = pkgdir(CalibrateEDMF)
    include(joinpath(src_dir, "parallel.jl"))
end

include(joinpath(cedmf, "driver", "global_parallel", "parallel_scm_eval.jl"))
