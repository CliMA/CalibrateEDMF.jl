"""Evaluates the stability of a set of SCM configurations for a parameter vector and possibly draws a new one."""

using CalibrateEDMF
cedmf = pkgdir(CalibrateEDMF)
include(joinpath(cedmf, "driver", "hpc_parallel", "precondition_prior.jl"))
