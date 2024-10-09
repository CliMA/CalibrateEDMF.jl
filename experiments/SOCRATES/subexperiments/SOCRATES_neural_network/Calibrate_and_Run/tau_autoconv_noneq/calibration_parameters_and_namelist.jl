#=
This file is meant to be included into a larger framework so intermediate variables may be undefined
=#

# ========================================================================================================================= #
# constants we use here
ρ_l = 1000. # density of ice, default from ClimaParameters
ρ_i = 916.7 # density of ice, default from ClimaParameters
r_0 = 20 * 1e-6 # 20 microns

# ========================================================================================================================= #
# NN stuff
nn_path = joinpath(experiment_dir, "Calibrate_and_Run", calibration_setup, "calibrate", "pretrained_NN.jld2")
nn_pretrained_params, nn_pretrained_repr, nn_pretrained_x_0_characteristic = JLD2.load(nn_path, "params", "re", "x_0_characteristic")
# ========================================================================================================================= #
expanded_unconstrained_σ = FT(1.0) # alternate for unconstrained_σ when we truly don't know the prior mean (or it's very uncertain)

calibration_parameters__experiment_setup = Dict( # The variables we wish to calibrate , these aren't in the namelist so we gotta add them to the local namelist...
    #
    "neural_microphysics_relaxation_network"   => Dict("prior_mean" => FT.(nn_pretrained_params)  , "constraints" => repeat([no_constraint()], length(nn_pretrained_params)) , "l2_reg" => nothing, "CLIMAParameters_longname" => nothing, "unconstrained_σ" => expanded_unconstrained_σ),  # have to use one FT throughout
    #
) 

local_namelist__experiment_setup = [ # things in namelist that otherwise wouldn't be... (both random parameters and parameters we want to calibrate that we added ourselves that generate_namelist doesn't insert...)
    ("user_aux", "neural_microphysics_relaxation_network", calibration_parameters__experiment_setup["neural_microphysics_relaxation_network"]["prior_mean"]),
    #
    ("user_aux", "model_re_location", nn_path), # i think nn_pretrained_repr is not isbits() so we can't use it in the namelist, so we use the path instead
    ("user_aux", "model_x_0_characteristic", FT.(nn_pretrained_x_0_characteristic)), # have to convert to FT for going in params etc...
    #
]
