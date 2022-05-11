# Installation instructions

CalibrateEDMF.jl is a registered Julia package. You can install the latest version
of CalibrateEDMF.jl through the built-in package manager. Initiate a julia session and run the commands

```julia
julia> ]
(v1.5) pkg> add CalibrateEDMF
(v1.5) pkg> instantiate
```

If you want to use the latest version of the package instead, clone the GitHub repository

```
> git clone https://github.com/CliMA/CalibrateEDMF.jl.git
```

In order to use the package, compile the project first.

```
> julia --project
julia> ]
(v1.5) pkg> instantiate
```

## Installation in dev mode (advanced)

Since both `TurbulenceConvection.jl` and `EnsembleKalmanProcesses.jl` are under rapid development, we may want to access a recent unpublished version of these packages when working with `CalibrateEDMF.jl`, or even use a version with local changes. If this is the case, clone the latest `EnsembleKalmanProcesses.jl` (resp. `EnsembleKalmanProcesses.jl`) version from GitHub (whichever you want to dev with),

```
> git clone https://github.com/CliMA/EnsembleKalmanProcesses.jl.git
> git clone https://github.com/CliMA/TurbulenceConvection.jl 
```

and try the following,

```
> julia --project
julia> ]
pkg> dev path/to/EnsembleKalmanProcesses.jl path/to/TurbulenceConvection.jl
pkg> instantiate
```

This will link CalibrateEDMF to your local version of `EnsembleKalmanProcesses.jl` (resp. `EnsembleKalmanProcesses.jl`), allowing rapid prototyping across packages.

If you want to use PyPlot with CalibrateEDMF.jl, and you are using PyPlot for the first time, just do

```
> julia --project
```

```julia
using Pkg
ENV["PYTHON"]=""
Pkg.build("PyCall")
exit()
```

And then compile,

```
> julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
```
