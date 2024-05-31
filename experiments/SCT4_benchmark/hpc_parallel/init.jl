"""Initializes a SCM calibration process."""

using CalibrateEDMF
cedmf = pkgdir(CalibrateEDMF)
include(joinpath(cedmf, "driver", "hpc_parallel", "init.jl"))
