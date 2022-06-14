using PackageCompiler
using Glob

using TurbulenceConvection
using CalibrateEDMF
using EnsembleKalmanProcesses
using OrdinaryDiffEq
using NCDatasets
using StochasticDiffEq
using ClimaCore

cedmf = pkgdir(CalibrateEDMF)

if isempty(Glob.glob("CEDMF.so"))
    create_sysimage(
        [
            "TurbulenceConvection",
            "EnsembleKalmanProcesses",
            "OrdinaryDiffEq",
            "NCDatasets",
            "StochasticDiffEq",
            "ClimaCore",
        ];
        sysimage_path = "CEDMF.so",
        precompile_execution_file = joinpath(cedmf, "test", "runtests.jl"),
    )
end