
# ========================================================================================================================= #
#  This is a template, you should use it in another script in which you set the following items and then include this file  #
# ========================================================================================================================= #

using Distributions
using StatsBase
using LinearAlgebra
using Random
# using CalibrateEDMF
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
FT = Float64

# Cases defined as structs for quick access to default configs
struct SOCRATES_Train end
struct SOCRATES_Val end

default_params = CalibrateEDMF.HelperFuncs.CP.create_toml_dict(FT; dict_type="alias") # name since we use the alias in this package, get a list of all default_params (including those not in driver/generate_namelist.jl) from ClimaParameters

wrap(x) = (x isa Union{Vector,Tuple}) ? x : [x] # wrap scalars in a vector so that they can be splatted regardless of what they are (then you can pass in eitehr vector or scalar in your calibration_parameters dict)



# ========================================================================================================================= #
# ========================================================================================================================= #
