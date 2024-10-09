# module TCRunnerUtils # I think we can't have a module that works with distributed, see  https://discourse.julialang.org/t/error-running-distributed-code-inside-of-a-module/54283/6
# maybe bring everything to a separate module except runTCs()
# export create_namelists_from_calibration_output,
#     get_calibrated_parameters,
#     run_TCs,
#     recursive_merge,
#     run_requested_parameters,
#     get_subdict_with_key

using Distributed
@everywhere begin
    using Pkg
    Pkg.activate(dirname(@__DIR__)) # i moved this so this should probalby move?
end
@everywhere begin
    # using CalibrateEDMF
    # using CalibrateEDMF.TurbulenceConvectionUtils
    # using CalibrateEDMF.HelperFuncs
    # pkg_dir = pkgdir(CalibrateEDMF)
    # import Logging # like TurbulenceConvectionUtils.jl
    using CalibrateEDMF.TCRunnerUtils
end
using Dates
# using ArgParse

# using TurbulenceConvection # also like TurbulenceConvectionUtils.jl to give us main1d
# tc = pkgdir(TurbulenceConvection)
# include(joinpath(tc, "driver", "main.jl")) # define the main.jl function...

# # these don't seem to be working inside the @everywhere block? idk why... race condition bug?
# using CalibrateEDMF
# pkg_dir = pkgdir(CalibrateEDMF)
# include(joinpath(pkg_dir, "tools", "DiagnosticsTools.jl"))

"""
For this file, I want to enable running any set of input cases (not just with the output from calibration)

We would like to enable
- getting a namelist from a calibration output
- running a set of case with a set of namelists
- outputing all the outputs in one place using a distributed framework

This would in principle allow us to do the sort of thing we did in TrainTau.jl, running grids of inputs, but with the additional functionality of using calibrated parameters and much easier slurm integration.
This is also good because we can take parameters from multiple different calibrations and combine them into one namelist and use it here

currently even merging dicts normally doesnt work bc stuff like config["turbulence"]["EDMF_PrognosticTKE"]["nn_ent_params"] don't match between runs

"""



function run_TCs(
    # namelists::Vector{Dict{<:Any, <:Any}}; # not sure how to specify vector of dicts... Vector{Dict} doesn't work, neihter does Vector{Dict{Any,Any}}
    namelists::Vector; # not sure how to specify vector of dicts... Vector{Dict} doesn't work, neihter does Vector{Dict{Any,Any}}
    tc_output_dir::Union{Nothing,String}="./", # the parent output dir for all the cases we run
    # config::Dict,
    # run_cases::Dict;
    # method::String = "best_nn_particle_mean",
    # metric::String = "mse_full",
    # n_ens::Int = 1,
)    
    # currently it's just one tc_output_dir but I gues youc ould expand that later... idk...


    @info "Preparing to run $(length(namelists)) forward model evaluations at $(Dates.now())"
    pmap(x->run_one_SCM(x; tc_output_dir=tc_output_dir), namelists) # seems not to get defined somehow... idk... maybe it's same problem as above w/ DiagnosticsTools.jl
    @info "Finished. Current time: $(Dates.now())"

end
# end # module