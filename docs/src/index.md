# CalibrateEDMF.jl

```@meta
CurrentModule = CalibrateEDMF
```

A package to calibrate atmospheric turbulence and convection parameterizations using gradient-free ensemble Kalman methods.

The rationale behind the calibration framework implemented in this package is thoroughly described in our preprint:

Lopez-Gomez, I., Christopoulos, C., Langeland Ervik, H. L., Dunbar, O. R. A., Cohen, Y., Schneider, T. (2022) **Training physics-based machine-learning parameterizations with gradient-free ensemble Kalman methods**. [preprint](https://doi.org/10.1002/essoar.10510937.1)

The manuscript also includes results for an extended eddy-diffusivity mass-flux (EDMF) closure of turbulence and convection trained using this package. The EDMF closure is implemented in [TurbulenceConvection.jl](https://github.com/CliMA/TurbulenceConvection.jl).

If you use this package for your own research, or find any of the ideas presented useful, consider citing our work.

## Authors

CalibrateEDMF.jl is being developed by the [Climate Modeling Alliance](https://clima.caltech.edu). The main developers are Ignacio Lopez-Gomez (lead), Haakon Ludvig Langeland Ervik, Charles Kawczynski, Costa Christopoulos and Yair Cohen.
