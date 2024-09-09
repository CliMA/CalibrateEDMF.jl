using CalibrateEmulateSample
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.DataContainers
import EnsembleKalmanProcesses as EKP
using JLD2
using NCDatasets
using Statistics
using Random 
using GaussianProcesses

include("../tools/DiagnosticsTools.jl")

const CES = CalibrateEmulateSample


exp_id = <experiment_id>
# linreg pre-calibration ekp object
eki_path = <ekp_subset_precalibration.jld2>
# linreg full calibration
output_dir_full =  <path/to/fullcalibration>
# linreg pre-calibration
output_dir = <path/to/precalibration>

iterations = (4, 8, 16)
num_params = 22
# save/load emulator
emulator_save_path = "emulator_gp_$exp_id.jld2"
save_emulator = true
load_emulator = false

randomly_select_N = nothing

@load eki_path Γ_i u_final obs_mean

prior_path = joinpath(output_dir, "prior.jld2")
prior = JLD2.load_object(prior_path)

diagnostics_path = joinpath(output_dir, "Diagnostics.nc")
ds = Dataset(diagnostics_path, "r")

diagnostics_path_full = joinpath(output_dir_full, "Diagnostics.nc")
ds_full = Dataset(diagnostics_path_full, "r")

phi_names_optimal_full, phi_optimal_full = optimal_parameters(diagnostics_path_full; method = "last_nn_particle_mean")

@assert phi_names_optimal_full == prior.name # ensure same parameter ordering

phi_names_optimal, phi_optimal = optimal_parameters(diagnostics_path; method = "last_nn_particle_mean")

@assert phi_names_optimal == prior.name # ensure same parameter ordering
@assert phi_names_optimal == phi_names_optimal_full


u_optimal = Emulators.transform_constrained_to_unconstrained(
    prior, phi_optimal
)
u_final = u_optimal


#### Emulate ####

function create_paired_data(ds, iterations, randomly_select_N = nothing, first_N = nothing)

    U = vcat([ds.group["particle_diags"]["u"][:, :, i] for i in iterations]...) # (num_iter * num_particles) x num_params
    G = vcat([ds.group["particle_diags"]["g"][:, :, i] for i in iterations]...) # (num_iter * num_particles) x (dof * num_cases)

    non_nan_indices = findall(row -> all(!isnan, row), eachrow(G))
    G = G[non_nan_indices, :]
    U = U[non_nan_indices, :]

    @assert size(U, 1) == size(G, 1)



    if !isnothing(first_N)
        G = G[:, 1:first_N]
    end

    # randomly select N samples
    if !isnothing(randomly_select_N)
        selected_indices = randperm(size(U, 1))[1:randomly_select_N]
        G = G[selected_indices, :]
        U = U[selected_indices, :]
    end



    factor_vector = maximum(G, dims=1)[:]

    input_output_pairs = PairedDataContainer(U', G', data_are_columns = true)
    return (input_output_pairs, factor_vector)
end


input_output_pairs, factor_vector = create_paired_data(ds, iterations, randomly_select_N)
unconstrained_inputs = CES.Utilities.get_inputs(input_output_pairs)


# Build emulator with data
if load_emulator && isfile(emulator_save_path)
    @load emulator_save_path emulator_gp
    println("Emulator loaded from $emulator_save_path")
else
    # If not loading from file, build and train the emulator
    gppackage = Emulators.SKLJL()
    gauss_proc = Emulators.GaussianProcess(gppackage, noise_learn = false)
    emulator_gp = Emulator(gauss_proc, 
                            input_output_pairs, 
                            normalize_inputs = true,  
                            standardize_outputs = true,
                            standardize_outputs_factors = factor_vector,
                            decorrelate = true, 
                            obs_noise_cov = Γ_i)
    optimize_hyperparameters!(emulator_gp)
    
    if save_emulator
        @save emulator_save_path emulator_gp
        println("Emulator saved to $emulator_save_path")
    end
end




#### Sample ####

println("initial parameters: ", u_final)

mcmc = MCMCWrapper(RWMHSampling(), obs_mean, prior, emulator_gp; init_params = u_final)
new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 1000, discard_initial = 0)

# begin MCMC
println("Begin MCMC - with step size ", new_step)
N_samples = 100_000
chain = MarkovChainMonteCarlo.sample(mcmc, N_samples; stepsize = new_step, discard_initial = 2_000)
display(chain)



# # Extract posterior samples
posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain);
constrained_posterior = Emulators.transform_unconstrained_to_constrained(
    prior, MarkovChainMonteCarlo.get_distribution(posterior)
)


# draw num samples from prior equal to MCMC N_samples
prior_samples_u = EKP.construct_initial_ensemble(prior, N_samples)
# transform to constrained space
prior_samples_u_phi = EKP.transform_unconstrained_to_constrained(prior, prior_samples_u)

constrained_prior = Dict()
for (i, name) in enumerate(prior.name)
    constrained_prior[name] = prior_samples_u_phi[i,:]
end



# Save output to NetCDF file
ds_out = Dataset("CES_mcmc_samples_$(exp_id).nc", "c")

# Add dimensions
defDim(ds_out, "param", num_params)
defDim(ds_out, "sample_prior", N_samples)
defDim(ds_out, "sample_posterior", N_samples)



# Save constrained_prior and constrained_posterior dictionaries
for (param, values) in constrained_prior
    varname = string("prior_", param)
    var = defVar(ds_out, varname, Float64, ("sample_prior",))
    var[:] = values
end

for (param, values) in constrained_posterior
    varname = string("posterior_", param)
    var = defVar(ds_out, varname, Float64, ("sample_posterior",))
    var[:] = values
end

phi_opt_var = defVar(ds_out, "phi_optimal", Float64, ("param",))
phi_opt_var[:] = phi_optimal

phi_opt_full_var = defVar(ds_out, "phi_optimal_full", Float64, ("param",))
phi_opt_full_var[:] = phi_optimal_full


param_names_var = defVar(ds_out, "param_names", String, ("param",))
param_names_var[:] = phi_names_optimal

close(ds_out)