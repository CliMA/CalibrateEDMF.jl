# CalibrateEDMF.jl

```@meta
CurrentModule = CalibrateEDMF
```

CalibrateEDMF is a julia package that enables the calibration of atmospheric turbulence and convection parameterizations using gradient-free ensemble Kalman methods. It provides a user-friendly framework to train parameterizations implemented in [TurbulenceConvection.jl](https://github.com/CliMA/TurbulenceConvection.jl), using the ensemble-based optimization methods implemented in [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl).

Some of the options enabled by the package are:

- Automatic regularization of parameter learning as an inverse problem,
- Minibatch training using EnsembleKalmanProcesses.jl,
- Kalman inversion with isotropic or anisotropic regularization,
- Tracking of validation diagnostics, given a user-specified validation dataset.

The rationale behind the calibration framework implemented in this package is thoroughly described in our paper:

Lopez-Gomez, I., Christopoulos, C., Langeland Ervik, H. L., Dunbar, O. R. A., Cohen, Y., Schneider, T. (2022) **Training physics-based machine-learning parameterizations with gradient-free ensemble Kalman methods**, *Journal of Advances in Modeling Earth Systems*, 14, e2022MS003105. [doi](https://doi.org/10.1029/2022MS003105)

The manuscript also includes results for an extended eddy-diffusivity mass-flux (EDMF) closure of turbulence and convection trained using this package. If you use this package for your own research, or find any of the ideas presented useful, please cite our work.

## Authors

CalibrateEDMF.jl is being developed by the [Climate Modeling Alliance](https://clima.caltech.edu). The main developers are Ignacio Lopez-Gomez (lead), Haakon Ludvig Langeland Ervik, Charles Kawczynski, Costa Christopoulos and Yair Cohen.
