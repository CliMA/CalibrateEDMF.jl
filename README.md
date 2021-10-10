## CalibrateEDMF.jl
A package to calibrate turbulence and convection parameterizations.

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
