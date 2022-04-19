# Pipeline

Module containing all high-level functions defining a training session workflow. User interactions should be done through the use of a `config.jl` file. See the `experiments` directory of the package for a few examples of `config.jl` files.

```@meta
CurrentModule = CalibrateEDMF.Pipeline
```

```@docs
init_calibration
restart_calibration
ek_update
update_validation
update_minibatch_inverse_problem
write_model_evaluators
init_diagnostics
update_diagnostics
```