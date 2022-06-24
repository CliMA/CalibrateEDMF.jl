"""Initializes a SCM calibration process."""

using CalibrateEDMF

cedmf = pkgdir(CalibrateEDMF)
include(joinpath(cedmf, "driver", "global_parallel", "init.jl"))
