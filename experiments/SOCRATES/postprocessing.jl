"""
We want to take our output runs and postprocess them to get the statistics we want.

Run simulations for:
    1. The last iteration ensemble
    2. The best iteration ensemble member (if it's not in the last iteration, otherwise just copy it over)
    3. The ensemble mean parameters of each iteration (or just the last one? idk...) -- we have the ensemble mean of all the individual ensemble runs already... 

For all these, copy over the loss/mse from Diagnostics.nc 
    - We wont have this for ensemble means in Diagnostic.nc, but we can calculate it the same way we did for the plots from the 
    - For the runs w/ ensemble mean paramters, we would need to invert norm_factors etc to get to unscaled space and calculate these values


We can store the full runs somewhere, but then we will want to take means over the SOCRATES_summary reference time period (do we want variances too?)
While doing this, we can calculate anything additional we want to save
    - e.g. derived microphyiscal process rates (from model and truth)
    - e.g. sedimentation rates (from model and from truth)
    - e.g. sublimation/deposition rates (from model and from truth)
    - e.g. combined liquid and ice category values

    We will also want to copy over the loss/mse from Diagnostics.nc (or can we use the norm_factors to go to constrained space and recalculate it?)

What do we do about failures lol?

We will have a dimensions called "role", where we'll have reference/calibrated as before I guess... initialize variables as NaN, fill in as we go along

"""



using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl")
using NCDatasets
using JSON
# using CalibrateEDMF # why is this so slow compared the later include? also breaks world age stuff...

FT = Float64

using ProgressMeter


CEDMF_dir = "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl"
CEDMF_data_dir = "/home/jbenjami/Data/Research_Schneider/CliMa/CalibrateEDMF.jl/"
experiments_dir = joinpath(CEDMF_dir, "experiments/SOCRATES")
# postprocess_dir = joinpath(experiments_dir, "Postprocessing") # eh we're gonna move this to be elsewhere... also we need data_storage
postprocess_dir = joinpath(CEDMF_dir, "experiments/SOCRATES")
postprocess_runs_storage_dir = joinpath(CEDMF_dir, "experiments/SOCRATES_postprocess_runs_storage") # store outside SOCRATES for rsync efficiency...
postprocess_runs_storage_data_dir = joinpath(CEDMF_data_dir, "experiments/SOCRATES_postprocess_runs_storage") # store outside SOCRATES for rsync efficiency...

"""
From https://discourse.julialang.org/t/how-to-include-into-local-scope/34634/11?u=jbphyswx
to include code by directly inserting it into the local scope, you can use a macro like this:
This hopefully will help solve world age problems arising from including config files (it didn't)
"""
macro safe_include(filename::AbstractString)
    path = joinpath(dirname(String(__source__.file)), filename)
    # @info "Including $path"
    return esc(Meta.parse("quote; " * read(path, String) * "; end").args[1])
end

# macro safe_include(filename::Symbol)
#     filename = String(eval(filename))
#     # convert filname to STring and call the macro
#     # @safe_include string(filename) is wrong bc that's an expression, needs to be evaluated
#     return esc(:(@safe_include $filename))
# end



# Underneath here keep the same structure as the experiments directory, subexperiments etc...




# include("postprocessing_variable_functions.jl") # methods for calculating derived quantities

new_calibration_vars_list = (
    ["ql_mean", "qi_mean"],
    ["ql_all_mean", "qi_all_mean"],
    ["ql_mean", "qi_mean", "qr_mean", "qip_mean"],
    ["temperature_mean", "ql_mean", "qi_mean"],
    ["temperature_mean", "ql_all_mean", "qi_all_mean"],
)

experiments = (
    "SOCRATES_Base",
    "SOCRATES_exponential_T_scaling_ice",
    "SOCRATES_exponential_T_scaling_ice_raw",
    "SOCRATES_powerlaw_T_scaling_ice",
    "SOCRATES_geometric_liq__geometric_ice",
    "SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice",
    "SOCRATES_geometric_liq__powerlaw_T_scaling_ice",
    "SOCRATES_neural_network",
    "SOCRATES_linear_combination",
    "SOCRATES_linear_combination_with_w",
)

setups = ("pow_icenuc_autoconv_eq", "tau_autoconv_noneq")

valid_experiment_setups = (
    ("SOCRATES_Base", "tau_autoconv_noneq"),
    ("SOCRATES_Base", "pow_icenuc_autoconv_eq"),
    ("SOCRATES_exponential_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_exponential_T_scaling_ice_raw", "tau_autoconv_noneq"),
    ("SOCRATES_powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__geometric_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__exponential_T_scaling_and_geometric_ice", "tau_autoconv_noneq"),
    ("SOCRATES_geometric_liq__powerlaw_T_scaling_ice", "tau_autoconv_noneq"),
    ("SOCRATES_neural_network", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination", "tau_autoconv_noneq"),
    ("SOCRATES_linear_combination_with_w", "tau_autoconv_noneq"),
)


# === Variable Processing ========================================================================================== #
neg = x -> -x
null_func = x -> x
g_to_kg = x -> x ./ FT(1000)
perday_to_persec = x -> x ./ FT(24 * 3600)
w_to_q = w -> w ./ (1 .+ w) # mixing ratio to specific humidity    
hPa_to_Pa = x -> x .* FT(100)

# The others would have more complicated rates...

#=
should process rates also use w_to_q???? how so given it's a derivative???

dq/dt = d/dt [w/(1+w)] =  dw/dt / (1+w)^2 --> dq/dt = dw/dt / (1+w)^2... so we'd need to multiply rates all by (1+w)^2 to get the right units... which is a more complicated fcn.. mostly should be close to the same but...
would make them all derived...


=#

all_needed_reference_vars_detailed = Dict{Tuple{String, String}, Tuple{Union{String, Nothing}, Function, Bool}}( # LES NAME, unit scaling func, already exists in TC,
    # `t` and `zf` or `zc` in group reference must exist for process_SOCRATES_Atlas_LES_reference() to work properly...
    ("t", "timeseries") => ("time", null_func, false), # don't need here
    ("t", "profiles") => ("time", null_func, false), # don't need here
    # ("zf", "profiles") => ("z", null_func, true), # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
    ("zc", "reference") => ("z", null_func, true), # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
    # ("p_f", "reference") => ("p", hPa_to_Pa, true), # assuming "p" is a reference pressure but sometimes maybe we should use real pressure "PRES" instead?
    ("p_c", "reference") => ("p", hPa_to_Pa, true), # assuming "p" is a reference pressure but sometimes maybe we should use real pressure "PRES" instead?
    ("ρ_c", "reference") => (nothing, null_func, true), # assuming "p" is a reference pressure but sometimes maybe we should use real pressure "PRES" instead? Hopefully these just fill w/ NaN on the reference side, need for p_mean, ρ_mean  
    ("ρ_f", "reference") => (nothing, null_func, false), # assuming "p" is a reference pressure but sometimes maybe we should use real pressure "PRES" instead? Hopefully these just fill w/ NaN on the reference side, need for p_mean, ρ_mean  

    #
    ("thetal_mean", "profiles") => ("THETAL", null_func, true),
    ("temperature_mean", "profiles") => ("TABS", null_func, true),
    #
    ("qt_mean", "profiles") => ("QT", w_to_q ∘ g_to_kg, true),
    ("ql_mean", "profiles") => ("QC", w_to_q ∘ g_to_kg, true), # Seems QCL and QC are the same
    ("qi_mean", "profiles") => ("QI", w_to_q ∘ g_to_kg, true), # seems QCI and QI are the same
    ("qr_mean", "profiles") => ("QR", w_to_q ∘ g_to_kg, true),
    ("qs_mean", "profiles") => ("QS", w_to_q ∘ g_to_kg, true),
    #
    ("qc_mean", "profiles") => ("QN", w_to_q ∘ g_to_kg, false), # cloud liquid and ice (not in TC.jl output)
    ("qg_mean", "profiles") => ("QG", w_to_q ∘ g_to_kg, false),
    ("qp_mean", "profiles") => ("QP", w_to_q ∘ g_to_kg, false), # total precipitation (rain + snow) , not in TC.jl output
    #
    ("p_mean", "profiles") => ("PRES", hPa_to_Pa, false), # need these to be in profiles but read from reference
    ("ρ_mean", "profiles") => ("RHO", null_func, false), # need these to be in profiles but read from reference
    #
    ("qi_mean_sub", "profiles") => ("PRD", perday_to_persec, false),
    ("qi_mean_dep", "profiles") => ("EPRD", perday_to_persec, false),
    ("ql_mean_cond_evap", "profiles") => ("PCC", perday_to_persec, true), # not sure this is right cause it says droplets
    #
    ("qi_mean_sed", "profiles") => ("QISED", perday_to_persec ∘ g_to_kg, true),
    # autoconv
    ("ql_mean_acnv", "profiles") => ("PRC", neg ∘ perday_to_persec, true),  # autoconversion QL to QR [QL -> QR]
    ("qi_mean_acnv_direct", "profiles") => ("PRCI", neg ∘ perday_to_persec, false), # autoconversion QI to QS [QI -> QS]
    ("qi_mean_acnv_thresh", "profiles") => ("PITOSN", neg ∘ perday_to_persec, false), # ice to snow due to threshold [QI -> QS]
    # accretion/collection
    ("ql_mean_accr_liq_rai", "profiles") => ("PRA", neg ∘ perday_to_persec, true),    # accretion QL by QR [QL -> QR]
    ("ql_mean_accr_liq_ice", "profiles") => ("PSACWI", neg ∘ perday_to_persec, true),    # accretion QL by QI [QL -> QI]
    ("ql_mean_accr_liq_gra_to_gra", "profiles") => ("PSACWG", neg ∘ perday_to_persec, false), # accretion QL by QG [QL -> QG] (https://doi.org/10.1029/2018JD028490 calls it collection)
    ("ql_mean_accr_liq_sno_to_sno", "profiles") => ("PSACWS", neg ∘ perday_to_persec, false), # accretion QL by QS [QG -> QS] ???? is this right
    ("ql_mean_accr_liq_sno_to_gra", "profiles") => ("PGSACW", neg ∘ perday_to_persec, false), # collection QL by QS, to QG
    ("qi_mean_accr_ice_liq", "profiles") => ("PSACWI", perday_to_persec, true), # accretion QL by QI [QL -> QI] (scaling shouldnt have a neg? on the ice direction since it's to ice not from ice?)
    ("qi_mean_accr_ice_rai_to_gra", "profiles") => ("PRACI", neg ∘ perday_to_persec, false), # collection QI by QR, to QG
    ("qi_mean_accr_ice_rai_to_sno", "profiles") => ("PRACIS", neg ∘ perday_to_persec, false), # collection QI by QR, to QG
    ("qi_mean_accr_ice_sno", "profiles") => ("PRAI", neg ∘ perday_to_persec, true), # accretion QI by QS [QI -> QS] (disagrees w/ https://doi.org/10.1029/2018JD028490 that calls it autoconv QI -> QS)

    # hetero ice
    ("qi_mean_het_nuc_immersion", "profiles") => ("MNUCCI", perday_to_persec, false), # Immersion freezing
    ("qi_mean_het_nuc_contact", "profiles") => ("MNUCCC", perday_to_persec, false), # Contact freezing
    # adv/sgs
    ("qi_mean_vert_adv", "profiles") => ("QIADV", perday_to_persec ∘ g_to_kg, true), # resolved vertical advection of ice (liq isn't in our files)
    ("qi_mean_ls_vert_adv", "profiles") => ("QILSADV", perday_to_persec ∘ g_to_kg, true), # large scale vertical ice advection (liq isn't in our files)
    ("qi_mean_sgs_tend", "profiles") => ("QIDIFF", perday_to_persec ∘ g_to_kg, true), # SGS ice flux (liq isn't in our files)
    ("qi_microphys", "profiles") => ("QIMPHY", perday_to_persec ∘ g_to_kg, true), # SGS ice flux (liq isn't in our files)
)

all_needed_reference_vars = Dict{Tuple{String, String}, Union{String, Nothing}}(
    key => value[1] for (key, value) in all_needed_reference_vars_detailed
)
var_LES_to_TC_scalings =
    Dict{Tuple{String, String}, Function}(key => value[2] for (key, value) in all_needed_reference_vars_detailed)

# Things that exist in TC and in LES
# TC_existing_to_LES_existing_translation = NTuple{12, Tuple{String, String}}(( #TC.jl (Name, Group)
# ("thetal_mean", "profiles"),
# ("temperature_mean", "profiles"),
# ("qt_mean", "profiles"),
# ("ql_mean", "profiles"),
# ("qi_mean", "profiles"),
# ("qr_mean", "profiles"),
# ("qs_mean", "profiles"),
# ("zf", "profiles"), # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
# ("zf", "reference"),# this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
# #
# ("ql_mean_cond_evap", "profiles"),
# ("ql_mean_sedimentation", "profiles"), # doesn't exist in LES, should be 0 in TC, no point I guess... idk maybe just leave it and let reference be NaN?
# ("qi_mean_sedimentation", "profiles"),
# ))
N_exist = sum(1 for (key, value) in all_needed_reference_vars_detailed if value[3])
TC_existing_to_LES_existing_translation =
    NTuple{N_exist, Tuple{String, String}}(key for (key, value) in all_needed_reference_vars_detailed if value[3])

# Things that don't exist in TC but do in LES (LES side)
# LES_existing_to_TC_new_translation_LES_side = Dict{Tuple{String,String}, String}( # TC.jl (Name, Group) -> fcn
# deprecated bc we'll need a TC name anyway to compare to TC output, instead we'll just use all_needed_reference_vars and just give it a TC name, and then create in TC w/ LES_existing_to_TC_new_translation
# )

function insert_dim(arr::AbstractArray, dim::Union{Int, Nothing} = nothing)
    # insert a dimension at dimension dim (default is last), 
    Nd = ndims(arr)
    if isnothing(dim)
        dim = Nd + 1
    end

    new_shape = size(arr)
    if dim > Nd + 1
        new_shape = (new_shape..., ones(Int, dim - Nd - 1)...)
    end

    new_shape = (new_shape[1:(dim - 1)]..., 1, new_shape[dim:end]...)
    return reshape(arr, new_shape)
end

function insert_dim(arr::AbstractArray, dim::Union{AbstractVector, Tuple} = [nothing])
    # insert a dimension at dimension dim, going down the list
    return foldl((a, d) -> insert_dim(a, d), dim; init = arr)
end



# Things that don't exist in TC but do in LES (TC side) that we want to recreate on TC side
LES_existing_to_TC_new_translation = Dict{Tuple{String, String}, Function}( # TC.jl (Name, Group) -> fcn
    ("qc_mean", "profiles") =>
        x -> Dict(
            "data" => x.group["profiles"]["ql_mean"] .+ x.group["profiles"]["qi_mean"],
            "dimnames" => NC.dimnames(x.group["profiles"]["ql_mean"]),
            "attrib" => nothing, # drop
        ),
    ("qp_mean", "profiles") =>
        x -> Dict(
            "data" => x.group["profiles"]["qr_mean"] .+ x.group["profiles"]["qs_mean"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qr_mean"]),
            "attrib" => nothing, # drop
        ),
    # no qg_mean bc although we need it for calculation on reference side, it can't be created in LES data
    ("p_mean", "profiles") =>
        x -> Dict(
            "data" => begin
                p_c = insert_dim(x.group["reference"]["p_c"], [4]) # (z × flight_number × role) -> (z, flight_number, role, method), ρ_c doesn't exist in LES data at all so we just make it NaN
                repeat(p_c, outer = size(x.group["profiles"]["temperature_mean"]) .÷ size(p_c))
            end, # 
            "dimnames" => NC.dimnames(x.group["profiles"]["temperature_mean"]), # copy from temp to expand to 2D
            "attrib" => nothing, # drop
        ),
    ("ρ_mean", "profiles") =>
        x -> Dict(
            "data" => begin
                ρ_c = insert_dim(x.group["reference"]["ρ_c"], [4])  # (z × flight_number × role) -> (z, flight_number, role, method), ρ_c doesn't exist in LES data at all so we just make it NaN
                repeat(ρ_c, outer = size(x.group["profiles"]["temperature_mean"]) .÷ size(ρ_c))
            end, # 
            "dimnames" => NC.dimnames(x.group["profiles"]["temperature_mean"]), # copy from temp to expand to 2D
            "attrib" => nothing, # drop
        ),
)

# Things that exist in TC but don't exist in LES (these are just calculations in TC lingo from existing data, we won't add anything new)
TC_existing_to_LES_new_translation = Dict{Tuple{String, String}, Function}( # TC.jl (Name, Group) =>  fcn
    ("ql_all_mean", "profiles") =>
        x -> Dict(
            "data" => x.group["profiles"]["ql_mean"] .+ x.group["profiles"]["qr_mean"],
            "dimnames" => NC.dimnames(x.group["profiles"]["ql_mean"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("qi_all_mean", "profiles") =>
        x -> Dict(
            "data" =>
                x.group["profiles"]["qi_mean"] .+ x.group["profiles"]["qs_mean"] .+ x.group["profiles"]["qg_mean"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("qip_mean", "profiles") =>
        x -> Dict(
            "data" => x.group["profiles"]["qs_mean"] .+ x.group["profiles"]["qg_mean"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("qi_mean_sub_dep", "profiles") =>
        x -> Dict(
            "data" => x.group["profiles"]["qi_mean_sub"] .+ x.group["profiles"]["qi_mean_dep"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean_sub"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("qi_mean_acnv", "profiles") =>
        x -> Dict(
            "data" => x.group["profiles"]["qi_mean_acnv_direct"] .+ x.group["profiles"]["qi_mean_acnv_thresh"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean_sub"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("ql_mean_accr_liq", "profiles") =>
        x -> Dict(
            "data" =>
                x.group["profiles"]["ql_mean_accr_liq_rai"] .+ x.group["profiles"]["ql_mean_accr_liq_gra_to_gra"] .+
                x.group["profiles"]["ql_mean_accr_liq_sno_to_sno"] .+
                x.group["profiles"]["ql_mean_accr_liq_sno_to_gra"] .+
                x.group["profiles"]["ql_mean_accr_liq_ice"],
            "dimnames" => NC.dimnames(x.group["profiles"]["ql_mean_accr_liq_rai"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("ql_mean_accr_liq_sno", "profiles") =>
        x -> Dict(
            "data" =>
                x.group["profiles"]["ql_mean_accr_liq_sno_to_sno"] .+
                x.group["profiles"]["ql_mean_accr_liq_sno_to_gra"],
            "dimnames" => NC.dimnames(x.group["profiles"]["ql_mean_accr_liq_sno_to_sno"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("qi_mean_accr_ice", "profiles") =>
        x -> Dict(
            "data" =>
                x.group["profiles"]["qi_mean_accr_ice_liq"] .+ x.group["profiles"]["qi_mean_accr_ice_rai_to_gra"] .+
                x.group["profiles"]["qi_mean_accr_ice_rai_to_sno"] .+
                x.group["profiles"]["qi_mean_accr_ice_sno"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean_accr_ice_liq"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("qi_mean_accr_ice_no_liq", "profiles") =>
        x -> Dict( # for saving bc we don't have liq/ice interaction (PSACWI) in TC...
            "data" =>
                x.group["profiles"]["qi_mean_accr_ice_rai_to_gra"] .+
                x.group["profiles"]["qi_mean_accr_ice_rai_to_sno"] .+
                x.group["profiles"]["qi_mean_accr_ice_sno"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean_accr_ice_liq"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("qi_mean_accr_ice_rai", "profiles") =>
        x -> Dict(
            "data" =>
                x.group["profiles"]["qi_mean_accr_ice_rai_to_gra"] .+
                x.group["profiles"]["qi_mean_accr_ice_rai_to_sno"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean_accr_ice_liq"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),
    ("qi_mean_het_nuc", "profiles") =>
        x -> Dict(
            "data" =>
                x.group["profiles"]["qi_mean_het_nuc_immersion"] .+ x.group["profiles"]["qi_mean_het_nuc_contact"],
            "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean_het_nuc_immersion"]),
            "attrib" => nothing, # drop attributes cause they'd be innacurate
        ),

    # ("qi_microphys", "profiles") => x-> Dict( # or could just use QIMPHY
    #         "data" => 
    #             x.group["profiles"]["qi_mean_sub"] .+
    #             x.group["profiles"]["qi_mean_dep"] .+
    #             x.group["profiles"]["qi_mean_acnv_direct"] .+
    #             x.group["profiles"]["qi_mean_acnv_thresh"] .+
    #             x.group["profiles"]["qi_mean_accr_ice_liq"] .+
    #             x.group["profiles"]["qi_mean_accr_ice_rai_to_gra"] .+
    #             x.group["profiles"]["qi_mean_accr_ice_rai_to_sno"] .+
    #             x.group["profiles"]["qi_mean_accr_ice_sno"] .+
    #             x.group["profiles"]["qi_mean_het_nuc_immersion"] .+
    #             x.group["profiles"]["qi_mean_het_nuc_contact"] .+
    #             x.group["profiles"]["qi_mean_sed"],
    #         "dimnames" => NC.dimnames(x.group["profiles"]["qi_mean_het_nuc_immersion"]),
    #         "attrib" => nothing # drop attributes cause they'd be innacurate
    #         ),

)

# ========================================================================================================================= #
#=
A problem we have with our current setup is that get_profile only reads straight from files, so everything needed for a calc needs to:
- the output of a calc must already be in the input file (e.g. how we put all the calculation precursors in the reference file)
- everything needed for the calc is already be part of the output so the calculation can be done in post (like how we do qc_mean bc ql_mean and qi_mean are already in the output)

If we wanted to have a function of variables that arent in the input file, e.g. say we wanted ql_var + qi_var in TC, we don't have those in the output so the calculation would have to be done in advance, forcing us to store it in refrence for truth, and who knows where for TC, so that our profiles fcn can find it in a file 
This attempts to prototype a sol'n that uses get_profile
- you'd still need to create the variable yourself in the output but then you could just fill it with the vector syou get from this :)
=#
TC_existing_to_LES_new_translation_alt = Dict{Tuple{String, String}, Function}(
    ("qc_mean", "profiles") =>
        x -> Dict(
            "data" =>
                (filename, config, reference_model, z_scm) ->
                    get_reference_period_TC_output(filename, ["ql_mean"], config, reference_model; z_scm = z_scm) .+ get_reference_period_TC_output(
                        filename,
                        ["qi_mean"],
                        config,
                        reference_model;
                        z_scm = z_scm,
                    ), # repeated calls to get_profile are slow so maybe we can optimize this...
            "dimnames" => NC.dimnames(x.group["profiles"]["ql_mean"]),
            "attrib" => nothing, # drop
        ),
)

# ========================================================================================================================= #

# deprecate this for now and just force yourself to add all the names you need to the reference file w/ LES names. This is better for writing fcns w/ dimnames, attrib etc...
# TC_existing_to_LES_new_translation_LES_names = Dict{Tuple{String,String}, Function}( # TC.jl (Name, Group) =>  fcn
# )

# Things that don't exist in TC or LES
# derived_data_vars_TC = Dict{Tuple{String,String}, Function}( # TC.jl (Name, Group) =>  fcn # -- these are done last and are all in TC lingo
# we don't have any of these yet
# ) 

# ========================================================================================================================= #
# Reference: Recreate w/ data_vars = all_needed_reference_vars, derived_data_vars = TC_existing_to_LES_new_translation

# Then, iterate over runs w/
# - TC_existing_to_LES_existing_translation : default, is on both sides
# - TC_existing_to_LES_new_translation: create only for LES
# - LES_existing_to_TC_new_translation: create only for TC

# Anything we'd need to create only in TC that's only in LES we have removed bc:
# - the inputs would need TC names already -- so far we've decided just to add them to the reference file... We could change this though.
# - however, these should still only be simple arithmetic calculations e.g. (+,-.*,/,^), 


# FIRST STEP: 
# - CREATE UP TO DATE REFERENCE FILE W/ DATA_VARS = ALL_NEEDED_REFERENCE_VARS, DERIVED_DATA_VARS = TC_EXISTING_TO_LES_NEW_TRANSLATION. NOW EVERYTHING SHOULD BE EXISTING IN BOTH TC AND REFERENCE FILE

# ========================================================================================================================= #






# run postprocessing runs
# run_postprocessing_runs = false # These are slurm jobs so do those separately
# if run_postprocessing_runs
#     include(joinpath(CEDMF_dir, "experiments/SOCRATES/launch_postprocess_runs.jl"))
# end


# construct output postprocessed files (can we do these as jobs that depend on the jobs above? -- would need to create child scripts for easy sbatch use)




default_methods = [
    "best_particle",
    "best_nn_particle_mean",
    "best_particle_final",
    "mean_best_ensemble",
    "mean_final_ensemble",
    "best_ensemble_mean",
]
# function to do the same time selection/averaging that CalibrateEDMF.jl did

"""
This would be something like run_SCM_handler in TurbulenceConvectionUtils.jl... can we just use that? but then we'd need param_map etc...
"""

thisdir = dirname(@__FILE__)
@info thisdir
function postprocess_run(
    CEDMF_output_dir::String,  # path to directory w/ Diagnostics.nc and config files etc
    save_dir::String, # path to directory to save postprocessed files
    # calibration_vars::Vector{String};
    ;
    # flight_string::String = "RFAll",
    # forcing_string::String = "Obs",
    # calibrate_to::String = "Atlas_LES",
    methods::Vector{String} = default_methods,
)
    # read the output data from the post processing run

    # pull the truth data

    # do translations and construct new file

    diagnostics_file = joinpath(CEDMF_output_dir, "Diagnostics.nc")
    config_file = joinpath(CEDMF_output_dir, "config.jl")

    ensemble_inds = NCDataset(diagnostics_file) do ds
        ds.group["particle_diags"]["particle"][:]
    end
    ensemble_size = length(ensemble_inds)


    # replace "best_ensemble_mean" in methods with "best_ensemble_mean_{$i}" for i in 1:ensemble_size
    methods = [
        method == "best_ensemble_mean" ? collect("best_ensemble_mean_{$i}" for i in 1:ensemble_size) : method for
        method in methods
    ]
    methods = reduce(vcat, methods, init = []) # init to ensue gets list out
    @info("Postprocessing methods:", methods)

    out_paths = [joinpath(save_dir, method) for method in methods]
    @info("Creating directories for postprocessed files:", out_paths)
    map(out_path -> mkpath(out_path), out_paths)


    # can we make this into a job array? would just need to store the above in a list or something right?
    # I'm also not sure these do any of the time averaging that we would want...
    # one downside is that this automatically runs the validation set... rn I have this set to the test set but we should check that's always want we want...
    n_proc_scm = 6 # number of processors to use for SCM (5 for SCM, 1 for control)
    for (i_m, method) in enumerate(methods)
        tc_output_dir = out_paths[i_m]
        log_dir = joinpath(tc_output_dir, "logs")
        data_dir = joinpath(tc_output_dir, "data")
        optimal_params_save_path = joinpath(tc_output_dir, "param_values.nc")
        mkpath(log_dir)
        mkpath(data_dir)
        run(
            `sbatch  -o $log_dir/%x.out  --parsable --partition=expansion --kill-on-invalid-dep=yes -n $n_proc_scm $thisdir/postprocessor.sbatch $CEDMF_output_dir $data_dir $optimal_params_save_path $method`,
        ) # no need for array yet
        # run(`sbatch  -o $log_dir/%x-%A_%a.out  --parsable --partition=expansion --kill-on-invalid-dep=yes $thisdir/postprocessor.sbatch $CEDMF_output_dir $tc_output_dir $method` )
    end

end

# ================================================================================================================================================================================================================================================== #
# ================================================================================================================================================================================================================================================== #
# ================================================================================================================================================================================================================================================== #


myquantile(A, p; dims, kwargs...) = mapslices(x -> quantile(x, p; kwargs...), A; dims) # https://github.com/JuliaStats/Statistics.jl/issues/23#issuecomment-1718706138 quantile isn't defined for multidimensional arrays for some reason

"""
could use TCRunner.jl (does it do any of the time averaging that we need?)

We already have optimal_parameters() for:
    - best_particle : particle with lowest mse in training (`metric` = "mse_full") or validation (`metric` = "mse_full_val") set.
    - best_nn_particle_mean : particle nearest to ensemble mean for the iteration with lowest mse.

but would like to add methods for:
    - best_particle_final : particle from final iteration with lowest mse in training (`metric` = "mse_full") or validation (`metric` = "mse_full_val") set.
    - mean_best_ensemble : new particle w/ mean parameters of the iteration with the lowest mse
    - mean_final_ensemble : new particle w/ mean parameters of the final iteration

it would also be cool to have:
    - best_ensemble_mean : iteration w/ ensemble mean with the lowest mse
    - best_ensemble_mean_{i}: ensemble member i from iteration w/ ensemble mean with the lowest mse
    #
    - best_ensemble_median : iteration w/ ensemble median with the lowest mse
    - best_ensemble_median_{i}: ensemble member i from iteration w/ ensemble median with the lowest mse
but that wouldn't make sense bc you wouldn't be able to re-run it, it's a mean of many runs... (how would we even store and run the parameters then?). for now I guess we'll stick to just looking at the overall plots and not having process rates...
"""


# Then, iterate over runs w/
# - TC_existing_to_LES_existing_translation : default, is on both sides
# - TC_existing_to_LES_new_translation: create only for LES, are already in TC
# - LES_existing_to_TC_new_translation: create only for TC, are already in LES

function collate_postprocess_runs(
    CEDMF_output_dir::String,  # path to directory w/ Diagnostics.nc and config files etc
    tc_output_dir::String,  # path to directory w/ Diagnostics.nc and config files etc
    save_dir::String, # path to directory to save postprocessed files
    ;
    # calibrate_to::String = "Atlas_LES",
    methods::Vector{String} = default_methods,
    all_needed_reference_vars::Dict{Tuple{String, String}, Union{String, Nothing}} = all_needed_reference_vars, # needed to create reference properly
    data_vars::Union{Vector{Tuple{String, String}}, NTuple{N, Tuple{String, String}} where N} = TC_existing_to_LES_existing_translation, # things that exist in both TC and reference file created from LES w/ all_needed_reference_vars
    derived_data_vars_LES::Dict{Tuple{String, String}, Function} = TC_existing_to_LES_new_translation, # things we need to create in reference file created from LES w/ all_needed_reference_vars
    derived_data_vars_TC::Dict{Tuple{String, String}, Function} = LES_existing_to_TC_new_translation, # things we need to create in TC that we already have in reference file created from LES w/ all_needed_reference_vars
    var_LES_to_TC_scalings::Dict{Tuple{String, String}, Function} = var_LES_to_TC_scalings, # how to scale the variables from LES to TC units
    reference_files_already_have_derived_data_vars_LES::Bool = false, # if you've already created the reference file w/ derived_data_vars_LES, you don't have to redo it...
    overwrite_reference::Bool = false,
    overwrite::Bool = false,
    delete_TC_output_files::Bool = false, # they take up a lot of space so once you've run the collation, may as well get rid of them right? they really slow down rsync too (to like hours)
)

    diagnostics_file = joinpath(CEDMF_output_dir, "Diagnostics.nc")
    config_file = joinpath(CEDMF_output_dir, "config.jl")

    # === Calculate New Reference Truth ====================================================================================== #
    include(config_file) # to get the truth paths etc... (though I think we actually need to recalculate our truth with all the variables we want... but we also need it to get the right reference period etc...)
    @info("config_file:", config_file)
    # @eval @safe_include $config_file # https://groups.google.com/g/julia-users/c/a_qReX-J7Ng/m/xM7X7diU8R8J

    # --------------------------------------------------------------------- # Doesn't work, probably bc eval goes back to global scope
    # copy config_file to another location and relplace any include(path) w/ @eval @safe_include $path
    # config_file_2 = joinpath(save_dir, "collated_data", "config_safe_include.jl")
    # # cp(config_file, config_file_2, force=true)
    # rm(config_file_2, force=true)
    # open(config_file_2, "w") do io
    #     for line in eachline(config_file, keep=true)
    #         if occursin("include", line)
    #             line = replace(line, "include(" => "@eval @safe_include \$(")
    #             @info("line:", line)
    #         end
    #         write(io, line)
    #     end
    # end
    # @info("config_file_2:", config_file_2)
    # @eval @safe_include $config_file_2 
    # --------------------------------------------------------------------- #

    @info("calibrate_to:", calibrate_to)
    @info("config_file:", config_file)
    @info("local_namelist", local_namelist)
    reference_files = reference_paths # was defined in config file

    @info(CalibrateEDMF)
    # config = get_config() # seems to fail w/ world age on first run idk... need a permanent fix for this
    config = @invokelatest get_config() # seems to fail w/ world age on first run idk... need a permanent fix for this... maybe this will work?
    ref_config = config["reference"] # really it's validation here but... we wanted them to be the same...
    namelist_args = @invokelatest CalibrateEDMF.HelperFuncs.get_entry(config["scm"], "namelist_args", nothing)
    kwargs_ref_model = @invokelatest CalibrateEDMF.ReferenceModels.get_ref_model_kwargs(
        ref_config;
        global_namelist_args = namelist_args,
    )
    reference_models = @invokelatest CalibrateEDMF.ReferenceModels.construct_reference_models(kwargs_ref_model)

    # ========================================================================================================================= #

    # get ensemble size
    ensemble_inds = NCDataset(diagnostics_file) do ds
        ds.group["particle_diags"]["particle"][:]
    end
    ensemble_size = length(ensemble_inds)

    # replace "best_ensemble_mean" in methods with "best_ensemble_mean_{$i}" for i in 1:ensemble_size
    original_methods = deepcopy(methods)
    methods = [
        method == "best_ensemble_mean" ? collect("best_ensemble_mean_{$i}" for i in 1:ensemble_size) : method for
        method in methods
    ]
    methods = reduce(vcat, methods)

    out_files = Dict()
    namelists = Dict()

    # get flight_numbers and forcing_types
    for method in methods
        out_files[method] = Dict()
        data_path = joinpath(tc_output_dir, method, "data")
        flight_dirs = readdir(data_path, join = true)
        for flight_dir in flight_dirs
            namelist = JSON.parsefile(joinpath(flight_dir, "namelist_SOCRATES.in"), dicttype = Dict, inttype = Int)
            flight_number = namelist["meta"]["flight_number"]
            forcing_type = Symbol(namelist["meta"]["forcing_type"])
            if !haskey(namelists, (flight_number, forcing_type))
                namelists[(flight_number, forcing_type)] = namelist
            end

            data_file = readdir(joinpath(flight_dir, "stats"), join = true)[1]
            data_file = joinpath(data_path, flight_dir, "stats", data_file)
            out_files[method][(flight_number, forcing_type)] = data_file

        end
    end
    forcing_types::Set{Symbol} = Set{Symbol}()
    flight_numbers::Set{Int} = Set{Int}()
    for method in methods
        for (flight_number, forcing_type) in keys(out_files[method])
            push!(forcing_types, forcing_type)
            push!(flight_numbers, flight_number)
        end
    end

    # create collated_data_dir
    """
    So we'll have:
        - groups: profiles, timeseries (timeseries empty rn)
        - dims: time, z, role, particle

    Each flight etc will remain in its own file, and we'll collate them w/ python later (can maybe save an output from that collation to avoid reading so many files repeatedly)
    """
    collated_data_dir = joinpath(save_dir, "collated_data")
    mkpath(collated_data_dir)
    collated_data_file = joinpath(collated_data_dir, "combined_data.nc")


    # === Recalculate reference files ====================================================================================== #
    # update the reference data so that we get everything we need in the reference file (do here cause need truth_dir from config file)
    reference_paths2 = @invokelatest process_SOCRATES_Atlas_LES_reference(;
        out_dir = truth_dir,
        truth_dir = truth_dir,
        data_vars = keys(all_needed_reference_vars),
        data_vars_rename = all_needed_reference_vars,
        derived_data_vars = derived_data_vars_LES,
        var_scalings = var_LES_to_TC_scalings,
        overwrite = overwrite_reference,
    ) # the folder where we store our truth (Atlas LES Data) # create trimmed data for calibration so the covarainces aren't NaN (failed anyway cause i think after interpolation NaNs can return because the non-NaN data don't form a contiguous rectangle :/ )
    reference_files = reference_paths2 # seem to need a different name for some reason.. idk... include was doing something weird to reference_paths
    reference_files_have_derived_data_vars_LES = false
    if overwrite_reference
        # derived_data_vars_LES = Dict{Tuple{String,String}, Function}() # nothing left to do here
    else
        @warn(
            "Assuming all LES derived data are alredy in the reference files, overwrite_reference is false, so they will not be calculated"
        )
    end
    # data_vars = (data_vars..., keys(derived_data_vars_LES)..., keys(derived_data_vars_TC)...)

    # ========================================================================================================================= #

    nvars = length(data_vars) + length(derived_data_vars_LES) + length(derived_data_vars_TC)
    all_data_vars::NTuple{nvars, Tuple{Tuple{String, String}, String}} = (
        [(data_var, "data_vars") for data_var in data_vars]...,
        [(data_var, "derived_data_vars_LES") for data_var in keys(derived_data_vars_LES)]...,
        [(data_var, "derived_data_vars_TC") for data_var in keys(derived_data_vars_TC)]...,
    )

    # all_data_vars::NTuple{length(data_vars), Tuple{Tuple{String,String}, String}} = ( [(data_var, "data_vars") for data_var in data_vars]...,) # test this bc we added derived_data_vars_LES to this when we updated reference file, derived_data_vars_TC we'll just do in post...


    groups::Tuple = ("profiles", "reference") # no timeseries for us cause we're only doing time means
    # ADD A GROUP FOR PARAMETERS!
    @warn(
        "Need to add a group for the parameters we calibrated -- I don't think we saved them so we could save them here with the data idk"
    )
    coords = Dict{String, Union{Nothing, Vector{String}, Vector{Int}, Vector{FT}}}(
        # "t" => nothing, we will only keep averages (and maybe variances?) over the reference period
        "z" => nothing, # will get from the data
        "flight_number" => sort(collect(flight_numbers)),
        "role" => ["reference", "calibrated"],
        "particle" => collect(1:ensemble_size),
        "method" => original_methods, # any _mean/_median will be filled in later from the ensemble
        # later we might add another dimension for ["mean, "variance"] over the reference period but not rn.
    )
    zs = Dict{Tuple{Int, Symbol}, Vector{FT}}() # store z values for each flight_number, forcing_type pair
    i_r::Int, i_c::Int = 1, 2 # index for  refernece nad calibrated (for nicer )
    # methods -- already have but drop the ones for each particle, we only need one since that will go down a dimension
    methods_no_ensemble = Tuple(method for method in original_methods if !occursin("best_ensemble_me", method))
    methods_ensemble = Tuple(method for method in original_methods if occursin("best_ensemble_me", method)) # get the methods that are tied to running an ensemble

    # we need to reduce file openings and closings somehow but idk how, it's really slowing everything to a crawl (esp w/ 100 ensemble members and this is w/ only one run_setup... maybe invokelatest is the problem?)
    local truth_data::Union{FT, Array{FT}}
    local var_data::Union{FT, Array{FT}}
    local namelist::Dict
    local reference_model::CalibrateEDMF.ReferenceModels.ReferenceModel
    local _new_data_sz::NTuple{N, Int} where {N}
    local _new_data::Array{FT}
    local _new_data_ens_sz::NTuple{N, Int} where {N}
    local _data_var_ens::String
    local particle_method::String
    local particle::Int
    local flight_number::Int
    local forcing_type::Symbol
    local reducer::Function

    local _new_data_sub_dev::Array{FT}
    local _new_data_add_dev::Array{FT}

    _new_dims::Dict{String, NTuple{N, String} where N} =
        Dict("profiles" => ("z", "flight_number", "role", "method"), "reference" => ("z", "flight_number", "role"))
    _new_dims_ens::Dict{String, NTuple{N, String} where N} =
        Dict("profiles" => ("z", "flight_number", "role", "particle"), "reference" => ("z", "flight_number", "role"))



    @info("forcing_types:", forcing_types)


    for forcing_type in forcing_types # this is technically not one of our dimensions...

        if !isfile(collated_data_file) || overwrite # only do if file doesn't exist or overwrite is true
            rm(collated_data_file, force = true) # NCDatasets claims it will overwrite file if it exists but it doesn't so we delete if it exists
            new_data = NCDataset(collated_data_file, "c")
            for group in groups
                NC.defGroup(new_data, group)
            end

            # get output z values (and pass this to get_profile? -- do the reference model z impact what we get out?)
            for flight_number in flight_numbers
                # get z (this is fine just to do once bc we're doing each flight separately...)
                if !haskey(zs, (flight_number, forcing_type))
                    namelist = namelists[(flight_number, forcing_type)]
                    case_name = namelist["meta"]["simname"]
                    config_ind = findfirst(x -> x == case_name, ref_config["case_name"])
                    reference_model = reference_models[config_ind]
                    zs[(flight_number, forcing_type)] = @invokelatest get_z_rectified_obs(reference_model) # changes w/ each flight _numbers
                end
            end
            # collate all the zs and get the sorted unique values out
            coords["z"] = sort(unique(vcat(values(Dict(k => v for (k, v) in zs if k[2] == forcing_type))...))) # should we change this to be from grid320 and grid 192 instead of from the runs themselves now that we change our z grid routinely?
            @info("coords z", coords["z"])

            # add coordinates we know (everything except time and z which we'll fill in at the end?)
            for group in ["profiles"]
                # dims
                NC.defDim(new_data.group[group], "z", length(coords["z"]))
                NC.defDim(new_data.group[group], "flight_number", length(coords["flight_number"]))
                NC.defDim(new_data.group[group], "role", length(coords["role"]))
                NC.defDim(new_data.group[group], "particle", length(coords["particle"]))
                NC.defDim(new_data.group[group], "method", length(coords["method"]))
                # coords
                NC.defVar(new_data.group[group], "z", coords["z"], ["z"])
                NC.defVar(new_data.group[group], "flight_number", coords["flight_number"], ["flight_number"])
                NC.defVar(new_data.group[group], "role", coords["role"], ["role"])
                NC.defVar(new_data.group[group], "particle", coords["particle"], ["particle"])
                NC.defVar(new_data.group[group], "method", coords["method"], ["method"])
            end

            for group in ["reference"]
                # dims
                NC.defDim(new_data.group[group], "z", length(coords["z"]))
                NC.defDim(new_data.group[group], "flight_number", length(coords["flight_number"]))
                NC.defDim(new_data.group[group], "role", length(coords["role"]))
                # coords
                NC.defVar(new_data.group[group], "z", coords["z"], ["z"])
                NC.defVar(new_data.group[group], "flight_number", coords["flight_number"], ["flight_number"])
                NC.defVar(new_data.group[group], "role", coords["role"], ["role"])
            end




            @info("all_data_vars: ", all_data_vars)

            # do the data processing
            @showprogress for (_vardef, varsource) in all_data_vars
                _data_var::String, _group::String = _vardef
                @assert _group ∈ groups



                _new_data_sz = Tuple(length(coords[dim]) for dim in _new_dims[_group])
                _new_data = fill(FT(NaN), _new_data_sz) # NaN in case of any failures? idk
                _new_data_ens_sz = Tuple(length(coords[dim]) for dim in _new_dims_ens[_group])

                ens_data_vars = Dict{Tuple{String, String, String}, Array{FT}}() # {name, group, method}

                # if _data_var == "ρ_c" # we already did this above
                #     @info("_data_var = ρ_c, group=$_group, _new_data_sz=$_new_data_sz")
                # end


                for (i_fn, flight_number) in enumerate(coords["flight_number"]) # looping over this first is nice bc we don't have to keep changing the reference model etc... but it's annoying for ensemble variables that are by method across flights, also creating variables is hard bc their dimensions are based on group...
                    # select truth, namelist, reference_model
                    truth_file = reference_files[(flight_number, forcing_type)]
                    namelist = namelists[(flight_number, forcing_type)]
                    case_name = namelist["meta"]["simname"]
                    config_ind = findfirst(x -> x == case_name, ref_config["case_name"])
                    reference_model = reference_models[config_ind]


                    # get truth (LES) data
                    if (varsource == "data_vars") ||
                       (varsource == "derived_data_vars_TC") ||
                       (varsource == "derived_data_vars_LES") # already is in both TC and LES ||  already is in LES reference but not TC || wasn't in LES but we calculated it in process_SOCRATES_Atlas_LES_reference hopefully
                        if nc_contains_var(truth_file, _data_var; group = _group)
                            # @info("ssstatus: ", truth_file, _data_var, config, reference_model, flight_number, _vardef, varsource)
                            truth_data = get_reference_period_TC_output(
                                truth_file,
                                [_data_var],
                                config,
                                reference_model;
                                z_scm = coords["z"],
                                has_time_dims = has_time_dims(truth_file, _data_var; group = _group),
                            ) # check if this is just a vector, in which case doing var by var is best and not combining
                        else
                            @warn("No variable $_data_var in truth file $truth_file, filling with NaN")
                            truth_data = FT(NaN)
                        end
                        #if ( varsource == "derived_data_vars_LES" ) technically, these should be in the reference file already bc we mean to do the rerun w/ all_needed_reference_vars and derived_data_vars_LES, so this if/ifelse could be collapsed
                        # alternately you could derive them, but bc we don't store the subvariables needed for the calc in this output, it's more challenging... 
                        # putting them in the reference_file w/ TC names is easiest... we could write separate functions to work on the original LES output files and variable names but we dont even have those filenames in this func, etc and the reference file is a 1 time calc permanent artifact...
                        # we should just be sure never to remove things from the reference that are needed elsewhere -- if this becomes annoying, we can store our reference files somewhere specific to this process...
                    end


                    if _group == "profiles"
                        for (i_m, method) in enumerate(coords["method"])
                            # @showprogress for (i_m, method) in enumerate(original_methods) # version w/ progress meter
                            if method ∉ methods_ensemble # these don't use the particle dimension
                                if (flight_number, forcing_type) ∉ keys(out_files[method])
                                    # @warn("No file for (method, flight_number, forcing_type): ($method, $flight_number, $forcing_type) for input directory $tc_output_dir. Skipping...")
                                else

                                    if (varsource == "data_vars") || (varsource == "derived_data_vars_LES") # derived_data_vars_TC we'll do later from full arrays, derived_data_vars_LES ones are already there
                                        data_file = out_files[method][(flight_number, forcing_type)]

                                        if nc_contains_var(data_file, _data_var; group = _group)
                                            var_data = get_reference_period_TC_output(
                                                data_file,
                                                [_data_var],
                                                config,
                                                reference_model;
                                                z_scm = coords["z"],
                                                has_time_dims = has_time_dims(truth_file, _data_var; group = _group),
                                            ) # check if this is just a vector, in which case doing var by var is best and not combining
                                        else
                                            @warn("No variable $_data_var in data file $data_file, filling with NaN")
                                            var_data = FT(NaN)
                                        end
                                        @inbounds _new_data[:, i_fn:i_fn, i_c:i_c, i_m:i_m] .= var_data
                                    end

                                    if (varsource == "data_vars") ||
                                       (varsource == "derived_data_vars_TC") ||
                                       (varsource == "derived_data_vars_LES")
                                        @inbounds _new_data[:, i_fn:i_fn, i_r:i_r, i_m:i_m] .= truth_data # should we just make these reference data? idk... maybe...
                                    end
                                end
                            else
                                _data_var_ens = "$(_data_var)_ensemble" # has to have a different name to have different coords...
                                if (_data_var_ens, _group, method) ∉ keys(ens_data_vars)
                                    ens_data_vars[(_data_var_ens, _group, method)] = fill(NaN, _new_data_ens_sz) # NaN in case of any failures? idk. Needs to be somewhere in here bc it's specific to this method, but needs to be shared across flight_numbers....
                                    # else
                                    # ens_data_vars[(_data_var_ens, _group, method)] = fill(NaN, _new_data_ens_sz) # NaN in case of any failures? idk. Needs to be somewhere in here bc it's specific to this method, but needs to be shared across flight_numbers....
                                end
                                _new_data_ens = ens_data_vars[(_data_var_ens, _group, method)]

                                _new_data_sub_dev = fill(FT(NaN), _new_data_sz) # storing uncertainty
                                _new_data_add_dev = fill(FT(NaN), _new_data_sz) # storing uncertainty

                                for particle in coords["particle"] # this slows everything wayyy down (opening and closing 100 files multiple times I guess...)
                                    # @showprogress for particle in coords["particle"]
                                    particle_method = "$(method)_{$particle}"

                                    if (flight_number, forcing_type) ∉ keys(out_files[particle_method])
                                        # @warn("No file for (method, flight_number, forcing_type): ($particle_method, $flight_number, $forcing_type) for input directory $tc_output_dir. Skipping...")
                                    else

                                        if (varsource == "data_vars") || (varsource == "derived_data_vars_LES") # derived_data_vars_TC we'll do later from full arrays, derived_data_vars_LES ones are already there
                                            data_file = out_files[particle_method][(flight_number, forcing_type)]
                                            if nc_contains_var(data_file, _data_var; group = _group)
                                                var_data = get_reference_period_TC_output(
                                                    data_file,
                                                    [_data_var],
                                                    config,
                                                    reference_model;
                                                    z_scm = coords["z"],
                                                    has_time_dims = has_time_dims(
                                                        truth_file,
                                                        _data_var;
                                                        group = _group,
                                                    ),
                                                )
                                            else
                                                # @warn("No variable $_data_var in data file $data_file, filling with NaN")
                                                var_data = FT(NaN)
                                            end

                                            @inbounds _new_data_ens[:, i_fn:i_fn, i_c:i_c, particle:particle] .=
                                                var_data
                                        end

                                        if (varsource == "data_vars") ||
                                           (varsource == "derived_data_vars_TC") ||
                                           (varsource == "derived_data_vars_LES")
                                            @inbounds _new_data_ens[:, i_fn:i_fn, i_r:i_r, particle:particle] .=
                                                truth_data
                                        end
                                    end
                                end
                                # add mean or median of ensemble to the data
                                if method == "best_ensemble_mean"
                                    reducer = mean
                                elseif method == "best_ensemble_median"
                                    reducer = median
                                else
                                    error("Method $method not recognized")
                                end
                                @inbounds _new_data[:, :, :, i_m:i_m] .=
                                    @invokelatest reducer(_new_data_ens[:, :, :, :], dims = 4) # not sure why this breaks even w/ safe_include but oh well...
                            end
                        end

                        # if _data_var == "ρ_mean"
                        #     @info "Processing $_data_var in $_group"
                        #     @info("truth_file: ", truth_file)
                        #     @info("truth_data: ", truth_data)
                        #     @info("var_data: ", var_data)
                        #     @info("new_data: ", _new_data)
                        # end

                    elseif _group == "reference" # reference data should be the same for all methods and the truth...
                        # data_file = truth_file
                        method = collect(keys(out_files))[1] # just need a method to get the data file, reference variables should not vary across methods
                        data_file = out_files[method][(flight_number, forcing_type)]
                        if nc_contains_var(data_file, _data_var; group = _group)
                            var_data = get_reference_period_TC_output(
                                data_file,
                                [_data_var],
                                config,
                                reference_model;
                                z_scm = coords["z"],
                                has_time_dims = has_time_dims(data_file, _data_var; group = _group),
                            )
                        else
                            @warn("No variable $_data_var in data file $data_file, filling with NaN")
                            var_data = FT(NaN)

                            # if _data_var == "ρ_c"
                            #     @info("_data_var = ρ_c, group=$_group, _new_data_sz=$_new_data_sz, data_")
                            # end
                        end

                        @inbounds _new_data[:, i_fn:i_fn, i_r:i_r] .= truth_data
                        @inbounds _new_data[:, i_fn:i_fn, i_c:i_c] .= var_data

                    end


                end

                # make nc vars
                NC.defVar(new_data.group[_group], _data_var, _new_data, _new_dims[_group];)

                # make ensemble data vars
                for ((_data_var_ens, _group, _), _new_data_ens) in ens_data_vars
                    NC.defVar(new_data.group[_group], _data_var_ens, _new_data_ens, _new_dims_ens[_group];) # ens vars get their own variable for each method? idk...
                end
            end




            # add derived variables to reference LES (that are already in TC) (all underlying needed vars should exist from above)
            # ThIS WON'T WORK BC THE VARIABLES NEEDED ARENT KEPT IN THE OUTPUT FILE -- we instead force the reference file to have them by calling process_SOCRATES_Atlas_LES_reference w/ all_needed_reference_vars and derived_data_vars_LES
            # for _vardef in derived_data_vars_LES
            #     if group == "profiles"
            #         _varname, _group = _vardef
            #         _new_data = derived_data_vars_LES[_vardef](_new_data)
            #         new_data.group[_group][_varname][:, i_r:i_r, :] = _new_data[:, i_r:i_r, :]
            #     end
            # end

            # add derived variables to TC (that are already in LES) (all underlying needed vars should exist from above)
            @warn(
                "This isn't true, if you need a variable from TC that has no equivalent in LES, you can't do this..... for now just put secondary needed variables as if they exist in the reference fail and let it fail and fill w/ NaNs there...?"
            )
            # -- To fix this, you could write our functions here to just take in outputs from that fcn though, like qc_mean having  data => get_reference_period_TC_output(data_file, ["ql",]) + get_reference_period_TC_output(data_file, ["qi",])... but that's a lot of work...
            # -- on the plus side, you wouldn't need this block to be at the end though and could do it inline w/ the other data vars...
            # -- you would also be passing in datafile instead of x being the netcdf data
            # -- you could also apply the same logic to the LES side and just have the functions take in the file and return the output fcn and that would greatly reduce the number of excess variables we need to store in combined_data.nc
            # THIS SHOULD WORK BC THE VARIABLES NEEDED ARE KEPT IN THE OUTPUT FILE (FOR NOW) -- LONG TERM WE NEED A WAY TO MIX GET_PROFILE AND THESE FUNCTIONS...
            for (_vardef, _varfunc) in derived_data_vars_TC
                _varname, _group = _vardef
                if _group == "profiles"
                    _varname, _group = _vardef
                    _new_data = _varfunc(new_data)["data"]
                    new_data.group[_group][_varname][:, :, i_c:i_c, :] = _new_data[:, :, i_c:i_c, :] # do only for calibrated side because that's TC where the variable didn't exist yet, the reference value should have been done in the reference processing
                    # do ensemble var # how do we do that with different names?
                    @warn(
                        "Need to add ensemble vars for derived_data_vars_TC, but they have _ensemble appended onto their names right so how do we do that..."
                    )
                elseif _group == "reference"
                    _varname, _group = _vardef
                    _new_data = _varfunc(new_data)["data"]
                    new_data.group[_group][_varname][:, :, i_c:i_c] = _new_data[:, :, i_c:i_c] # do only for calibrated side, the reference side gets done in reference processing rn, and we have to have created this var already to be overwriting it...
                    # do ensemble var # how do we do that with different names?
                    @warn(
                        "Need to add ensemble vars for derived_data_vars_TC, but they have _ensemble appended onto their names right so how do we do that..."
                    )
                end
            end

            # derived variables on both sides (from things we already have put into the output for reference and TC)
            # none for now

            close(new_data)
        end
    end


    # delete the things in TC_output_dir to save space/transfer time (if this becomes a problem we could just store them somewhere else...)
    if delete_TC_output_files
        for method in methods
            ##  (deprecated cause we've moved to already storing them outside the postprocess dir)
            # old_path = joinpath(tc_output_dir, method)
            # replace SOCRATES w/ SOCRATES_postprocess_runs_storage (deprecated cause we've moved to already storing them outside the postprocess dir)
            # new_path = replace(old_path, "SOCRATES" => "SOCRATES_postprocess_runs_storag/Data_Storage", count=1) # replace first occurence, (so not the SOCRATES in subexperiment)
            # mkpath(new_path)
            # run(`mv $old_path $new_path`) # do this instead of rm(old_path, force=true) just in case

            ## straight up delete...
            old_path = joinpath(tc_output_dir, method)
            rm(old_path, force = true)
        end
    end

end

"""
It's slow to iterate over all the particles (the current iteration at least) so we're making this slurm version to launch separate jobs
But for now it only work's w/ the default options cause idk how to pass dicts etc...
"""
function collate_postprocess_runs_slurm(
    CEDMF_output_dir::String,  # path to directory w/ Diagnostics.nc and config files etc
    tc_output_dir::String,  # path to directory w/ Diagnostics.nc and config files etc
    save_dir::String, # path to directory to save postprocessed files
    ;
    # calibrate_to::String = "Atlas_LES",
    methods = default_methods,
    data_vars = TC_existing_to_LES_existing_translation, # test
    derived_data_vars = TC_existing_to_LES_new_translation, # test
    #
    reference_files_already_have_derived_data_vars_LES::Bool = false, # if you've already created the reference file w/ derived_data_vars_LES, you don't have to redo it...
    overwrite_reference::Bool = false,
    overwrite::Bool = false,
    delete_TC_output_files::Bool = false, # they take up a lot of space so once you've run the collation, may as well get rid of them right? they really slow down rsync too (to like hours)
)

    mkpath(joinpath(save_dir, "collated_data")) # otherwise slurm will fail silently
    run(
        `sbatch  -o $save_dir/collated_data/%x.out  --parsable --partition=expansion --kill-on-invalid-dep=yes $thisdir/collate_postprocess_runs.sbatch $CEDMF_output_dir $tc_output_dir $save_dir $overwrite $overwrite_reference $reference_files_already_have_derived_data_vars_LES $delete_TC_output_files`,
    ) # no need for array yet

end




"""
Use the reference model to get the reference period data for the simulation at data_file

We use a ReferenceModel bc they're cheap to construct and give us the time shifts and windowing from our config for each flight number for free
We also get any z truncations as well.
"""
function get_reference_period_TC_output(
    data_file::String,
    y_names::Vector{String}, # the variables we want
    config::Dict, # this would come from the config file being included and then running get_config()
    reference_model;
    z_scm::Union{Nothing, Vector{FT}} = nothing, # if we already have z_scm we can pass it in
    penalized_value::FT = FT(NaN), # value to fill in for penalized values
    verbose::Bool = false,
    has_time_dims::Union{Bool, Vector{Bool}} = true,
)
    if isnothing(z_scm)
        z_scm::Vector{FT} = @invokelatest CalibrateEDMF.ReferenceModels.get_z_rectified_obs(reference_model)
    end

    # use reference model bc we get the time shifts and windowing from config for free
    # @info("status: ", reference_model, data_file, y_names, z_scm, penalized_value, verbose)
    return @invokelatest CalibrateEDMF.ReferenceStats.get_profile(
        reference_model,
        data_file,
        y_names;
        z_scm = z_scm,
        prof_ind = false,
        penalized_value = penalized_value,
        verbose = verbose,
        has_time_dims = has_time_dims,
    ) # does this concatenate them, if so maybe easier to go one variable at a time? idk... Also idk if it returns our z...


    # If we can create a faster version of this w/o the TC fluff that could be good, and also allow us to use functions on vars instead of just var names

end


function has_time_dims(
    nc_data::Union{String, NCDatasets.NCDataset},
    varname::String;
    group::Union{String, Nothing} = nothing,
    time_dim = "time",
)

    nc_data = isa(nc_data, String) ? NCDatasets.NCDataset(nc_data) : nc_data

    if isnothing(group)
        return time_dim ∈ NC.dimnames(nc_data[varname])
    else
        return time_dim ∈ NC.dimnames(nc_data.group[group][varname])
    end

end



function nc_contains_var(
    nc_data::Union{String, NCDatasets.NCDataset},
    varname::String;
    group::Union{String, Nothing} = nothing,
)

    nc_data = isa(nc_data, String) ? NCDatasets.NCDataset(nc_data) : nc_data

    if isnothing(group)
        return varname in keys(nc_data)
    else
        return varname in keys(nc_data.group[group])
    end

end
