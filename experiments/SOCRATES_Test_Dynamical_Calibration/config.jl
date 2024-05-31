#= Custom calibration configuration file. =#

using Distributions
using StatsBase
using LinearAlgebra
using Random
using CalibrateEDMF
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.DistributionUtils
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.LESUtils
using CalibrateEDMF.TurbulenceConvectionUtils
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.HelperFuncs
# Import EKP modules
using JLD2
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const NC = CalibrateEDMF.NetCDFIO.NC # is already loaded so let's not reload

# Cases defined as structs for quick access to default configs
struct SOCRATES_Train end
struct SOCRATES_Val end
this_dir = @__DIR__ # the location of this file
data_dir = joinpath(this_dir, "Truth") # the folder where we store our truth (Atlas LES Data)
flight_numbers = [1,9,10,11,12,13]
forcing_types  = [:obs_data, :ERA5_data]

function get_config()
    config = Dict()
    # Flags for saving output data
    config["output"] = get_output_config()
    # Define regularization of inverse problem
    config["regularization"] = get_regularization_config()
    # Define reference used in the inverse problem 
    config["reference"] = get_reference_config(SOCRATES_Train())
    # Define reference used for validation
    config["validation"] = get_reference_config(SOCRATES_Val())
    # Define the parameter priors
    config["prior"] = get_prior_config()
    # Define the kalman process
    config["process"] = get_process_config()
    # Define the SCM static configuration
    config["scm"] = get_scm_config()
    return config
end

function get_output_config()
    config = Dict()
    config["outdir_root"] = pwd()
    return config
end

function get_regularization_config()
    config = Dict()
    # Regularization of observations: mean and covariance
    config["perform_PCA"] = true # Performs PCA on data
    config["variance_loss"] = 1.0e-2 # Variance truncation level in PCA
    config["normalize"] = true  # whether to normalize data by pooled variance
    config["tikhonov_mode"] = "relative" # Tikhonov regularization
    config["tikhonov_noise"] = 1.0e-6 # Tikhonov regularization
    config["dim_scaling"] = true # Dimensional scaling of the loss

    # Parameter regularization: L2 regularization with respect to prior mean.
    #  - Set to `nothing` to use prior covariance as regularizer,
    #  - Set to a float for isotropic parameter regularization.
    #  - Pass a dictionary of lists similar to config["prior_mean"] for
    #       anisotropic regularization. The dictionary must be complete.
    #       If you want to avoid regularizing a certain parameter, set the entry
    #       to [0].
    # To turn off regularization, set config["process"]["augmented"] to false.
    #
    #
    # Defaults set to batch_size/total_size to match total dataset uncertainty
    # in UKI. Feel free to set treat these as hyperparameters.
    config["l2_reg"] = Dict(
        # entrainment parameters
        "nn_ent_params" => repeat([0.0], 58),
        "turbulent_entrainment_factor" => [5.0 / 60.0],

        # diffusion parameters
        "tke_ed_coeff" => [5.0 / 60.0],
        "tke_diss_coeff" => [5.0 / 60.0],
        "static_stab_coeff" => [5.0 / 60.0],
        "tke_surf_scale" => [5.0 / 60.0],
        "Prandtl_number_0" => [5.0 / 60.0],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [5.0 / 60.0],
        "pressure_normalmode_buoy_coeff1" => [5.0 / 60.0],
        "pressure_normalmode_drag_coeff" => [5.0 / 60.0],

        # surface
        "surface_area" => [5.0 / 60.0],
    )
    return config
end

function get_process_config()
    config = Dict()
    config["N_iter"] = 50
    config["N_ens"] = 50 # Must be 2p+1 when algorithm is "Unscented"
    config["algorithm"] = "Inversion" # "Sampler", "Unscented", "Inversion"
    config["noisy_obs"] = false # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0
    # Artificial time stepper of the EKI.
    config["Δt"] = 60.0 / 5.0 # 1.0
    # Whether to augment the outputs with the parameters for regularization
    config["augmented"] = true
    config["failure_handler"] = "sample_succ_gauss" #"high_loss" #"sample_succ_gauss"
    return config
end

function get_reference_config(::SOCRATES_Train)
    config = Dict()

    # Setup SOCRATES run arguments I guess
    flight_numbers = (9,)
    aux_kwargs     = (,) # fill in later
    # append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])
    
    # need reference dirs from wherever i put my truth, maybe add a SOCRATESUtils to match les_utils etc.
    n_repeat = length(ref_dirs)
    config["case_name"] = repeat(["SOCRATES"], n_repeat)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = SOCRATES()
    config["Σ_reference_type"] = SOCRATES()
    config["y_names"] =
    config["y_dir"] = ref_dirs
    config["t_start"] = repeat([0], n_repeat)
    config["t_end"] = repeat([14.0 * 3600], n_repeat)
    # Use full  timeseries for covariance (for us we just use what)?
    config["Σ_t_start"] = repeat([-5.75 * 24 * 3600], n_repeat) # might need to discard beginning for spinup etc..
    config["Σ_t_end"] = repeat([6.0 * 3600], n_repeat)
    config["batch_size"] = 5 # 
    config["write_full_stats"] = false
    return config
end

function get_reference_config(::SOCRATES_Val)
    config = Dict()

    # Train on same thing? or waht do we do here        repeat([["s_mean", "ql_mean", "qt_mean", "total_flux_qt", "total_flux_s", "u_mean", "v_mean"]], n_repeat) # are these what we want to validate on? so theta_li, qt, for our dynamical calibration?

    flight_numbers = (9,) # just validate on same flight
    aux_kwargs     = (,) # fill in later
    # ref_dirs = [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers]

    n_repeat = length(ref_dirs)
    config["case_name"] = repeat(["SOCRATES"], n_repeat)
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    config["y_reference_type"] = SOCRATES()
    config["Σ_reference_type"] = SOCRATES()
    config["y_names"] =
        repeat([["s_mean", "ql_mean", "qt_mean", "total_flux_qt", "total_flux_s", "u_mean", "v_mean"]], n_repeat) # change here to qt, theta_li? (remove s for now or use theta_l for now since we have that)
    config["y_dir"] = ref_dirs
    config["t_start"] = repeat([0.0 * 3600], n_repeat)
    config["t_end"] = repeat([14.0 * 3600], n_repeat)
    # Use full LES timeseries for covariance (we will change to use what here, a timeseries from the les data? or what)
    # config["Σ_t_start"] = repeat([-5.75 * 24 * 3600], n_repeat)
    # config["Σ_t_end"] = repeat([6.0 * 3600], n_repeat)
    config["write_full_stats"] = false
    return config
end

function get_prior_config()
    config = Dict()
    config["constraints"] = Dict(
        # entrainment parameters
        # "nn_ent_params" => [repeat([no_constraint()], 58)...], # remove
        # need to add entrainment parameters for moisture deficit clousure
        "turbulent_entrainment_factor" => [bounded(0.0, 10.0)],

        # diffusion parameters
        "tke_ed_coeff" => [bounded(0.01, 1.0)],
        "tke_diss_coeff" => [bounded(0.01, 1.0)],
        "static_stab_coeff" => [bounded(0.01, 1.0)],
        "tke_surf_scale" => [bounded(1.0, 16.0)],
        "Prandtl_number_0" => [bounded(0.5, 1.5)],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [bounded(0.0, 100.0)],
        "pressure_normalmode_buoy_coeff1" => [bounded(0.0, 10.0)],
        "pressure_normalmode_drag_coeff" => [bounded(0.0, 50.0)],

        # surface
        "surface_area" => [bounded(0.01, 0.5)],
    )

    # TC.jl prior mean
    config["prior_mean"] = Dict(
        # entrainment parameters
        "nn_ent_params" => 0.1 .* (rand(58) .- 0.5),
        "turbulent_entrainment_factor" => [0.075],

        # diffusion parameters
        "tke_ed_coeff" => [0.14],
        "tke_diss_coeff" => [0.22],
        "static_stab_coeff" => [0.4],
        "tke_surf_scale" => [3.75],
        "Prandtl_number_0" => [0.74],

        # momentum exchange parameters
        "pressure_normalmode_adv_coeff" => [0.001],
        "pressure_normalmode_buoy_coeff1" => [0.12],
        "pressure_normalmode_drag_coeff" => [10.0],

        # surface
        "surface_area" => [0.1],
    )

    config["unconstrained_σ"] = 1.0
    # Tight initial prior for Unscented
    # config["unconstrained_σ"] = 0.25
    return config
end

function get_scm_config() # set all my namelist stuff here
    config = Dict()
    config["namelist_args"] = [
        ("time_stepping", "dt_min", 1.0),
        ("time_stepping", "dt_max", 2.0),
        ("stats_io", "frequency", 60.0),
        ("turbulence", "EDMF_PrognosticTKE", "entrainment", "None"),
        ("turbulence", "EDMF_PrognosticTKE", "ml_entrainment", "NN"),
        ("turbulence", "EDMF_PrognosticTKE", "area_limiter_power", 0.0),
        ("turbulence", "EDMF_PrognosticTKE", "entr_dim_scale", "inv_z"),
        ("turbulence", "EDMF_PrognosticTKE", "detr_dim_scale", "inv_z"),
    ]
    return config
end

# The variables we need for calibration routine from the Atlas LES outputs
default_data_vars = Dict{Tuple{String,String}, String}( #TC.jl (Name, Group) =>  Atlas Name
("thetal_mean", "profiles") => "THETAL",
("temperature_mean", "profiles") => "TABS",
("qt_mean", "profiles") => "QT",
("t", "timeseries") => "time",
("t", "profiles") => "time",
("zf", "profiles") => "z", # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
("zf", "reference") => "z", # this is how it works in TC.jl, do we need to add "zc" separately? (note "zf" though in TC.jl has an extra 0 point at surface...) # I dont think we need this one though cause HelperFuncs.jl pairs them together in get_height anyway, might need it for is_face_variable() in HelperFuncs.jl though...
)

"""
Function to take the Atlas LES output and convert it to the format the calibration expects
 - subset
 - change names

 # each <flight_number, forcing_type> pair has it's own directory since the model expects the ref_dir to only contain one flight
"""
function process_SOCRATES_truth(;
    flight_numbers::Vector{Int} = flight_numbers,
    forcing_types::Vector{Symbol} = forcing_types,
    data_vars = keys(default_data_vars), # the variables we need...
    data_vars_rename::Dict{Tuple{String,String}, String} = default_data_vars,
    )
    truth_files = filter(contains(".nc",), readdir(data_dir))

    for forcing_type in forcing_types
        forcing_type_short =  replace.(string.(forcing_type), "_data" => "")  # just the ERA5 or Obs
        for flight_number in flight_numbers
            truth_file = filter(x -> contains("RF"*string(flight_number,pad=2))(x) & contains(lowercase(forcing_type_short))(lowercase(x)), truth_files)
            if ~isempty(truth_file) # if we have a file
                truth_file = truth_file[] # get the item out
                truth_file = joinpath(data_dir, truth_file)
                # Load the truth data
                truth_data = NC.Dataset(truth_file,"r")
                # apparently in julia there's no way to subset and work on a dataset in memory only so from here on we gotta goto disk already...
                name = "RF"*string(flight_number,pad=2)*"_"*string(forcing_type)
                outpath = joinpath(data_dir, name, "stats",name*".nc") # the format CalibrateEDMF.jl expects from TurbulenceConvection.jl
                mkpath(dirname(outpath)) # make the directory if it doesn't exist
                rm(outpath,force=true) # NCDatasets claims it will overwrite file if it exists but it doesn't so we delete if it exists
                new_data = NC.Dataset(outpath, "c") # create the new dataset
                # add the required groups for interplay w/ TC.jl
                for group in ["profiles", "reference", "timeseries"]
                    NC.defGroup(new_data, group)
                end
                # add dimensions
                for _dim in keys(truth_data.dim)
                    NC.defDim(new_data, _dim, truth_data.dim[_dim]) # truth_data.dim[_dim] gives the size
                end

                # add attributes
                for _attrib in keys(truth_data.attrib)
                    Base.setindex!(new_data.attrib, truth_data.attrib[_attrib], _attrib)
                end
                # add variables & rename
                for _vardef in data_vars
                    _data_var, _group = _vardef
                    _truth_var = data_vars_rename[_vardef]
                    _new_dims = collect((x=="z" ? "zf" : x for x in NC.dimnames(truth_data[_truth_var]))) # replace any "z" in dimnames with "zf"
                    NC.defVar(new_data.group[_group], _data_var, truth_data[_truth_var][:], _new_dims; attrib =  truth_data[_truth_var].attrib )
                end

                # Calculated and Fixed Variables
                # -- Need to calculate and add ("zc", "profiles"), and ("zc", "reference"), grid shouldn't have to be the same as TC.jl and it should interpolate using vertical_interpolation() from HelperFuncs.jl, if so we need to add z=0 to z/zf and also something at the surface for variables defined on z/zf, the latter part being the hard part lol so I hope we don't have to do that...
                zf_data = new_data.group["profiles"]["zf"][:]
                zc_data = (zf_data[1:end-1] .+ zf_data[2:end]) ./ 2.0
                _new_dims  =  collect((x=="z" ? "zc" : x for x in NC.dimnames(truth_data["z"]))) # replace any "z" in dimnames with "zc"
                NC.defVar(new_data.group["profiles"], "zc", zc_data, (_new_dims); attrib =  truth_data["z"].attrib )
                NC.defVar(new_data.group["reference"], "zc", zc_data, _new_dims; attrib =  truth_data["z"].attrib )
                # -- Need to fix time variable to go from days to second
                new_data.group["profiles"]["t"][:] = (new_data.group["profiles"]["t"][:] .- new_data.group["profiles"]["t"][1]) .* (24.0 .* 3600.0) |> (x -> x .+ (ceil(x[end]/3600)*3600 - x[end])) # shift day to second, then i think t=0 is missing and we need to shift it to end at t=12,14, otherwise would end at 11.91,13.91h so shift to nearest hour
                new_data.group["timeseries"]["t"][:] = (new_data.group["timeseries"]["t"][:] .- new_data.group["timeseries"]["t"][1]) .* (24.0 .* 3600.0 ) |> (x -> x .+ (ceil(x[end]/3600)*3600 - x[end])) # shift day to second, then i think t=0 is missing and we need to shift it to end at t=12,14, otherwise would end at 11.91,13.91h so shift to nearest hour
                #close file at end
                close(new_data)
            else
                @info("Skiping missing file for flight $flight_number and forcing $forcing_type")
            end
        end
    end
end

