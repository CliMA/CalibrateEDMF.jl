## CalibrateEDMF.jl

A package to calibrate turbulence and convection parameterizations.

|||
|---------------------:|:----------------------------------------------|
| **Docs Build**       | [![docs build][docs-bld-img]][docs-bld-url]   |
| **Documentation**    | [![dev][docs-dev-img]][docs-dev-url]          |
| **GHA CI**           | [![gha ci][gha-ci-img]][gha-ci-url]           |
| **Code Coverage**    | [![codecov][codecov-img]][codecov-url]        |
| **Bors enabled**     | [![bors][bors-img]][bors-url]                 |

[docs-bld-img]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/docs.yml/badge.svg
[docs-bld-url]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/docs.yml

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/CalibrateEDMF.jl/dev/

[gha-ci-img]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/ci.yml/badge.svg
[gha-ci-url]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/ci.yml

[codecov-img]: https://codecov.io/gh/CliMA/CalibrateEDMF.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/CalibrateEDMF.jl

[bors-img]: https://bors.tech/images/badge_small.svg
[bors-url]: https://app.bors.tech/repositories/37644

# Installation

To use the package, clone this repository

  >> git clone https://github.com/CliMA/CalibrateEDMF.jl.git

Before calibration, we need to compile the project.

>> julia --project

>> julia> ]

>> pkg> instantiate

# Installation in dev mode

Since both `TurbulenceConvection.jl` and `EnsembleKalmanProcesses.jl` are under rapid development, we may want to access a recent unpublished version of these packages when working with `CalibrateEDMF.jl`, or even use a version with local changes. If this is the case, clone the latest `EnsembleKalmanProcesses.jl` (resp. `EnsembleKalmanProcesses.jl`) version from GitHub (whichever you want to dev with),

  >> git clone https://github.com/CliMA/EnsembleKalmanProcesses.jl.git

  >> git clone https://github.com/CliMA/TurbulenceConvection.jl 

and try the following,

>> julia --project

>> julia> ]

>> pkg> dev path/to/EnsembleKalmanProcesses.jl path/to/TurbulenceConvection.jl

>> pkg> instantiate

This will link CalibrateEDMF to your local version of `EnsembleKalmanProcesses.jl` (resp. `EnsembleKalmanProcesses.jl`), allowing rapid prototyping across packages.

If you want to use PyPlot with CalibrateEDMF.jl, and you are using PyPlot for the first time, just do

>> julia --project

```
using Pkg
ENV["PYTHON"]=""
Pkg.build("PyCall")
exit()
```

And then compile,

>> julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
