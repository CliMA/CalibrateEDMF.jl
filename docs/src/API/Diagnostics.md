# Diagnostics

```@meta
CurrentModule = CalibrateEDMF.Diagnostics
```

This module defines and implements diagnostics of the training process. Diagnostics are organized into groups (defined as dictionaries `io_dictionary_...`) and written to file in netCDF4 format. All implemented dictionaries, as well as all their entries, are defined below. These include error metrics, parameter ensembles and their statistical moments, validation diagnostics, and more. 

The diagnostics are meant to evolve organically as users require more information regarding the training process. If you can think of diagnostics that may be useful to a wide range of users, consider opening an issue or pull request on [GitHub](https://github.com/CliMA/CalibrateEDMF.jl).

Data within the resulting `Diagnostics.nc` file can be analyzed using the Julia package [NCDatasets.jl](https://alexander-barth.github.io/NCDatasets.jl/stable/). The file can also be processed using software written in other popular programming languages like python through [netCDF4-python](http://unidata.github.io/netcdf4-python/) or [xarray](https://docs.xarray.dev/en/stable/).

```@docs
io_dictionary_metrics
io_dictionary_val_metrics
io_dictionary_ensemble
io_dictionary_particle_state
io_dictionary_particle_eval
io_dictionary_val_particle_eval
io_dictionary_reference
io_dictionary_val_reference
io_dictionary_prior
get_ϕ_cov
get_metric_var
get_mean_nearest_neighbor
```
