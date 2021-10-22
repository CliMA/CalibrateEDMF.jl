## CalibrateEDMF.jl

A package to calibrate turbulence and convection parameterizations.


|||
|---------------------:|:----------------------------------------------|
| **Docs Build**       | [![docs build][docs-bld-img]][docs-bld-url]   |
| **Documentation**    | [![dev][docs-dev-img]][docs-dev-url]          |
| **GHA CI**           | [![gha ci][gha-ci-img]][gha-ci-url]           |
| **Bors enabled**     | [![bors][bors-img]][bors-url]                 |

[docs-bld-img]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/docs.yml/badge.svg
[docs-bld-url]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/docs.yml

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/CalibrateEDMF.jl/dev/

[gha-ci-img]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/ci.yml/badge.svg
[gha-ci-url]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/ci.yml

[bors-img]: https://bors.tech/images/badge_small.svg
[bors-url]: https://app.bors.tech/repositories/37644

# Installation

To use the package, clone this repo and the `TurbulenceConvection.jl` repo from GitHub.

  >> git clone https://github.com/CliMA/CalibrateEDMF.jl.git

  >> git clone https://github.com/CliMA/TurbulenceConvection.jl 

Before calibration, we need to compile the project. For the time being `TurbulenceConvection.jl` is not a published package, so this dependency is added manually by

>> julia --project

>> julia> ]

>> pkg> dev path/to/TurbulenceConvection.jl

>> pkg> instantiate

Since both `TurbulenceConvection.jl` and `EnsembleKalmanProcesses.jl` are under rapid development, the latest version of `TurbulenceConvection.jl` may not be compatible with the published version of `EnsembleKalmanProcesses.jl`. If this is the case, clone the latest `EnsembleKalmanProcesses.jl` version from GitHub and try the following,

>> julia --project

>> julia> ]

>> pkg> dev path/to/EnsembleKalmanProcesses.jl path/to/TurbulenceConvection.jl

>> pkg> instantiate

CalibrateEDMF.jl also depends on PyPlot. If you are using PyPlot for the first time, just do

>> julia --project

```
using Pkg
ENV["PYTHON"]=""
Pkg.build("PyCall")
exit()
```

And then compile,

>> julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
