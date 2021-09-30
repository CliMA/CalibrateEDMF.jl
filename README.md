## CalibrateEDMF.jl: A Package to calibrate turbulence and convection parameterizations

Last updated : September 2021

# Installation

TO use the package, clone this repo and the `TurbulenceConvection.jl` repo from GitHub.

  >> git clone https://github.com/CliMA/CalibrateEDMF.jl.git

  >> git clone https://github.com/CliMA/TurbulenceConvection.jl 

Before calibration, we need to compile the project. Note that each experiment in CalibrateEDMF.jl is its own project, so we will have to perform compilation within each example in the `experiments` directory. For the time being `TurbulenceConvection.jl` is not a published package, so the dependencies are added manually by

>> julia --project

>> julia> ]

>> pkg> dev path/to/TurbulenceConvection.jl

>> pkg> instantiate

Since both `TurbulenceConvection.jl` and `EnsembleKalmanProcesses.jl` are under rapid development, the latest version of `TurbulenceConvection.jl` may not be compatible with the published version of `EnsembleKalmanProcesses.jl`. If this is the case, clone the latest `EnsembleKalmanProcesses.jl` version from GitHub and try the following,

>> julia --project

>> julia> ]

>> pkg> dev path/to/EnsembleKalmanProcesses.jl path/to/TurbulenceConvection.jl

>> pkg> instantiate
