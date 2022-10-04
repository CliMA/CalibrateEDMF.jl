## CalibrateEDMF.jl

A package to calibrate atmospheric turbulence and convection parameterizations using gradient-free ensemble Kalman methods.

The rationale behind the calibration framework implemented in this package is thoroughly described in our paper:

Lopez-Gomez, I., Christopoulos, C., Langeland Ervik, H. L., Dunbar, O. R. A., Cohen, Y., Schneider, T. (2022) **Training physics-based machine-learning parameterizations with gradient-free ensemble Kalman methods**, *Journal of Advances in Modeling Earth Systems*, 14(8), e2022MS003105. [doi](https://doi.org/10.1029/2022MS003105)

If you use this package for your own research, or find any of the ideas presented useful, please cite our work. The article also includes results for an extended eddy-diffusivity mass-flux (EDMF) closure of turbulence and convection trained using this package. The EDMF scheme is implemented in [TurbulenceConvection.jl](https://github.com/CliMA/TurbulenceConvection.jl), and described in the following papers:

Cohen, Y., Lopez-Gomez, I., Jaruga, A., He, J., Kaul, C., and Schneider, T. (2020) **Unified entrainment and detrainment closures for extended eddy-diffusivity mass-flux schemes.** *Journal of Advances in Modeling Earth Systems*, 12, e2020MS002162. [doi](https://doi.org/10.1029/2020MS002162)

Lopez-Gomez, I., Cohen, Y., He, J., Jaruga, A., Schneider, T. (2020) **A generalized mixing length closure for eddy-diﬀusivity mass-flux schemes of turbulence and convection.** *Journal of Advances in Modeling Earth Systems*, 12, e2020MS002161. [doi](https://doi.org/10.1029/2020MS002161)

For further details on how to use CalibrateEDMF, consult the [documentation](https://CliMA.github.io/CalibrateEDMF.jl/dev/).



|||
|---------------------:|:----------------------------------------------|
| **Documentation**    | [![dev][docs-latest-img]][docs-latest-url]    |
| **DOI**              | [![DOI][zenodo-img]][zenodo-latest-url]       |
| **Docs Build**       | [![docs build][docs-bld-img]][docs-bld-url]   |
| **GHA CI**           | [![gha ci][gha-ci-img]][gha-ci-url]           |
| **Code Coverage**    | [![codecov][codecov-img]][codecov-url]        |
| **Bors enabled**     | [![bors][bors-img]][bors-url]                 |

[zenodo-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.6382864.svg
[zenodo-latest-url]: https://doi.org/10.5281/zenodo.6382864

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://CliMA.github.io/CalibrateEDMF.jl/dev/

[docs-bld-img]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/docs.yml/badge.svg
[docs-bld-url]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/docs.yml

[gha-ci-img]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/ci.yml/badge.svg
[gha-ci-url]: https://github.com/CliMA/CalibrateEDMF.jl/actions/workflows/ci.yml

[codecov-img]: https://codecov.io/gh/CliMA/CalibrateEDMF.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/CalibrateEDMF.jl

[bors-img]: https://bors.tech/images/badge_small.svg
[bors-url]: https://app.bors.tech/repositories/37644

### Requirements

Julia version 1.5+ 

# Installation

To use latest version of this package, clone this repository

  >> git clone https://github.com/CliMA/CalibrateEDMF.jl.git

To use the latest stable release, you can install the package on your Julia environment:

  >> julia
  >> julia> using Pkg; Pkg.add("CalibrateEDMF")

In order to use the package, compile the project first.

>> julia --project

>> julia> ]

>> pkg> instantiate

# Installation in dev mode (advanced)

Since both `TurbulenceConvection.jl` and `EnsembleKalmanProcesses.jl` are under rapid development, we may want to access a recent unpublished version of these packages when working with `CalibrateEDMF.jl`, or even use a version with local changes. If this is the case, clone the latest `EnsembleKalmanProcesses.jl` (resp. `EnsembleKalmanProcesses.jl`) version from GitHub (whichever you want to dev with),

  >> git clone https://github.com/CliMA/EnsembleKalmanProcesses.jl.git

  >> git clone https://github.com/CliMA/TurbulenceConvection.jl 

and try the following,

>> julia --project

>> julia> ]

>> pkg> dev path/to/EnsembleKalmanProcesses.jl path/to/TurbulenceConvection.jl

>> pkg> instantiate

This will link CalibrateEDMF to your local version of `EnsembleKalmanProcesses.jl` (resp. `EnsembleKalmanProcesses.jl`), allowing rapid prototyping across packages.
