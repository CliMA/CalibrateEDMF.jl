## Performing a grid search and calculating loss maps over pairs of parameters for the TurbulenceConvection.jl EDMF

Authors: Haakon Ludvig Langeland Ervik, Yair Cohen

Last updated : April 2022

This tool relies on a `config.jl` used in a calbration process. To run this, the following sub-dictionary needs to be added to the `config.jl` file:
```
function get_grid_search_config()
    config = Dict()
    # Grid search is performed over each pair of parameters, across all specified values
    config[“parameters”] = Dict(
        “entrainment_factor” => range(0.05, 0.3, length=11),
        “detrainment_factor” => range(0.45, 0.6, length=11),
        “sorting_power” => range(0.0, 0.4, length=5),
    )
    # Number of simulations to run with identical configuration (except random seed)
    config["ensemble_size"] = 1
    # grid search output data stored in `<output_root_dir>/output/YYmmdd_abc`
    config["output_root_dir"] = pwd()
    # Perform grid search and loss map calculation for either the `reference` or `validation` set.
    config["sim_type"] = "reference"  # `reference` or `validation`
    return config
end
```
additionally, add the following line to the `get_config()` function in `config.jl`:
```
# grid search configuration
config["grid_search"] = get_grid_search_config()
```

In the grid search config, 
- `parameters` is a dictionary that specifies the parameters that will be used to generate 2D loss maps
for each pair of parameters, along with the values for each parameter.
- `ensemble_size` is an integer which specifies the number of times a specific simulation configuration is repeated (typically useful for stochastic models)
- `output_root_dir` specifies the parent directory to a directory called `output`, in which you may store all your grid search output. A specific grid search run is stored in a folder identified by today's date + a random three-letter string `YYmmdd_abc`. The full output path is therefore `<output_root_dir>/output/YYmmdd_abc`.
- `sim_type` selects which dataset to use for the map: choose `"reference"` to run a grid search over the `reference` configurations, or `"validation"` to run a grid search over the validation configurations.

Running a grid search generates an output folder in which the raw `TurbulenceConvection.jl` data is stored in a systematic way, as well as the `config.jl` file used to generate it.
If you are on the Caltech Central Cluster, you can run grid search by adding it to the schedule:

  >> sbatch run_grid_search.sbatch path/to/config.jl

In case some folders are missing due to an incomplete grid search, you can restart the grid search by

  >> sbatch restart_grid_search.sbatch path/to/output/YYmmdd_abc

where `YYmmdd_abc` is the directory for the grid search you want to restart.

The loss map can be generated by:

  >> sbatch run_loss_map.sbatch path/to/output/YYmmdd_abc

This works even if some simulations are missing. Missing simulations are replaced by `NaN` in the loss map. The loss map is stored in a NetCDF4 file named `loss.nc` in the `YYmmdd_abc` output folder.

To quickly generate rudimentary plots of the loss map, run

  >> julia --project quick_plots --sim_dir path/to/output/YYmmdd_abc

To generate loss map corner plots, see [VizCalibrateEDMF](https://github.com/CliMA/VizCalibrateEDMF).
