#=
This file is meant to be included into a larger framework so intermediate variables may be undefined
=#

expanded_unconstrained_σ = FT(2.5) # alternate for unconstrained_σ when we truly don't know the prior mean (or it's very uncertain)  ## TESTING 100 HERE!!!! (seem to have some nan errors...) (bounded_below values are log spaced  so consider that)
calibration_parameters__experiment_setup = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    #
    "τ_cond_evap" => Dict("prior_mean" => global_param_defaults["τ_cond_evap"]["prior_mean"], "constraints" => global_param_defaults["τ_cond_evap"]["constraints"] , "l2_reg" => nothing, "CLIMAParameters_longname" => "condensation_evaporation_timescale", "unconstrained_σ" => global_param_defaults["τ_cond_evap"]["unconstrained_σ"]), 
    #
    "τ_sub_dep" => Dict("prior_mean" => global_param_defaults["τ_sub_dep"]["prior_mean"], "constraints" => global_param_defaults["τ_sub_dep"]["constraints"] , "l2_reg" => nothing, "CLIMAParameters_longname" => "sublimation_deposition_timescale", "unconstrained_σ" => global_param_defaults["τ_sub_dep"]["unconstrained_σ"]),
)

# global local_namelist = [] # i think if you use something like local_namelist = ... below inside the function it will just create a new local variable and not change this one, so we need to use global (i think we didnt need after switching to local_namelist_here but idk...)
local_namelist__experiment_setup = [ # things in namelist that otherwise wouldn't be... (both random parameters and parameters we want to calibrate that we added ourselves that generate_namelist doesn't insert...)
    #
    ("microphysics", "τ_sub_dep"  , calibration_parameters__experiment_setup["τ_sub_dep"  ]["prior_mean"] ),
    ("microphysics", "τ_cond_evap", calibration_parameters__experiment_setup["τ_cond_evap"]["prior_mean"] ),
    #
]