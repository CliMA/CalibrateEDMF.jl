"""Performs a single step of the EnsembleKalmanProcess."""

using CalibrateEDMF

cedmf = pkgdir(CalibrateEDMF)
include(joinpath(cedmf, "driver", "global_parallel", "step_ekp.jl"))
