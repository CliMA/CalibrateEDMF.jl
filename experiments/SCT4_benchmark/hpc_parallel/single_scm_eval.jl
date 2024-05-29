"""Evaluates a set of SCM configurations for a single parameter vector."""

using CalibrateEDMF
cedmf = pkgdir(CalibrateEDMF)
include(joinpath(cedmf, "driver", "hpc_parallel", "single_scm_eval.jl"))
