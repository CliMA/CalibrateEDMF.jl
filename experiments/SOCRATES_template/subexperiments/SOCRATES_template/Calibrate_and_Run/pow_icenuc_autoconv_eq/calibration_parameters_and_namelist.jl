#=
This file is meant to be included into a larger framework so intermediate variables may be undefined
=#


calibration_parameters__experiment_setup = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    #
    "pow_icenuc" => Dict("prior_mean" => global_param_defaults["pow_icenuc"]["prior_mean"], "constraints" => global_param_defaults["pow_icenuc"]["constraints"] , "l2_reg" => nothing, "CLIMAParameters_longname" => "pow_icenuc", "unconstrained_σ" => global_param_defaults["pow_icenuc"]["unconstrained_σ"]),
    #
    ) # these aren't in the default_namelist so where should I put them?

# global local_namelist = [] # i think if you use something like local_namelist = ... below inside the function it will just create a new local variable and not change this one, so we need to use global (i think we didnt need after switching to local_namelist_here but idk...)
local_namelist__experiment_setup = [ # things in namelist that otherwise wouldn't be... (both random parameters and parameters we want to calibrate that we added ourselves that generate_namelist doesn't insert...)
    #
    ("microphysics", "pow_icenuc"  , calibration_parameters__experiment_setup["pow_icenuc"]["prior_mean"] ), # You'd think it was thermodynamics, but I think Parameters.jl TCP.thermodynamics_params sources from microphsyics... 
    #
]


