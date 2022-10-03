# Perfect model setup for optimal design experiments

*Author(s): Ignacio Lopez-Gomez*

The objective of this tutorial is to use the iterative solution to an inverse problem provided by an ensemble Kalman process to generate training data for a UQ emulator. We can create a well-posed perfect model inverse problem setup with the EDMF scheme following the steps outlined below.

## Perfect model data generation

- Define the parameters ϕ at which the perfect model will be set. This can be done randomly, or by calibrating the model to LES. These parameters ϕ are used to generate the data `y`, and to validate our results. During calibration (i.e., training sample generation) we assume we do not know the value of ϕ.

- Generate the perfect model data using the model with parameters ϕ, y=G(ϕ). To produce cfSite simulations, we will use the TCRunner tool. `SCT_run.sbatch` is an example script that takes the parameters ϕ from the last iteration of a calibration process, and generates the data defined by a data generation config file (in this case `hadgem_amip_sct_config.jl`). Check `SCT_run.sbatch` for recommended use instructions.

- An example of the generated data can be found at `/groups/esm/ilopezgo/optimal_design/perf_model_HadGEM2_nz55_B38_Inv_d40_p2_ewR` on the central cluster.

## Noise definition

We have generated the observations/data `y` from the perfect model. However, the EDMF implementation with steady forcing has very little noise and does not account for model error. To define a better-posed problem, we use the LES internal variability to define the noise.

This means that in our config we will use

```julia
config["y_reference_type"] = SCM()
config["Σ_reference_type"] = LES()
```

## Prior definition

We also need to define the parameters ϕ̃ at which we set the prior mode/mean. Remember that we cannot use the knowledge of ϕ at this point. For now, I have taken them to be the default parameters in TC.jl, which were calibrated in Cohen et al. (2020) and Lopez-Gomez et al. (2020).

## Example config

An example config file used to generate forward model evaluations following an ensemble Kalman process is `config_lwp_amip_stretched_2p_PERFECT.jl`. In this case we generate evaluations of liquid water path in all AMIP cfSites of Shen et al (2022) using the HadGEM2-A model. The parameters to be learned are the entrainment and detrainment coefficients.

## Output diagnostics

The output diagnostics of this process can be used as training samples for the emulator. An example is in the directory `results_Inversion_dt_1.0_p2_e20_i10_mb_SCM_2022-09-27_16-46_UhM`.
