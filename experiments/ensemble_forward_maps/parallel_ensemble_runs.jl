@everywhere include("ensemble_run.jl")

# Run ensemble for every var_params entry
@everywhere params = [
    "sde"
    # "sde_entr_std", "sde_detr_std",
    # "sde_entr_theta", "sde_detr_theta", 
    # "sde_entr_mu", "sde_detr_mu", 
    # "entr_lognormal_var", "detr_lognormal_var"
]
@everywhere val(x::Real) = repeat([x], length(params))
@everywhere var_params = [
    val(0.0),
]
n_ens=40
@everywhere f_(x) = run_ensemble($params, x, $n_ens)

map(f_, var_params)
