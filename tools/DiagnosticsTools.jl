using NCDatasets

"""
    Find indices of particle with lowest mse.
"""
function particle_diags_min_index(ds::NCDataset; metric::String = "mse_full")
    mse_mat = ds.group["particle_diags"][metric][:] # (n_particles, n_iterations)
    replace!(mse_mat, NaN => Inf)
    part_min_i, iter_min_i = Tuple(argmin(mse_mat))
    return part_min_i, iter_min_i
end

"""
    optimal_parameters(ds_path::String; method::String = "best_nn_particle_mean", metric::String = "mse_full")
Given path to Diagnostics.nc file, return optimal parameter set across all iterations and corresponding parameter names.

Inputs: 
 - `ds_path`: path to Diagnostics.nc file
 - `method`: method for computing optimal parameters. Returns parameters of:
        "best_particle" - particle with lowest mse in training (`metric` = "mse_full") or validation (`metric` = "val_mse_full") set.
        "best_nn_particle_mean" - particle nearest to ensemble mean for the iteration with lowest mse.
 - `metric` : mse metric to find the minimum of {"mse_full", "val_mse_full"}.
Returns: 
    - `u_names`: vector of parameter names
    - `u`: vector of optimal parameter values
"""

function optimal_parameters(ds_path::String; method::String = "best_nn_particle_mean", metric::String = "mse_full")

    valid_metrics = ("mse_full", "val_mse_full")
    @assert metric in valid_metrics "Invalid metric: $metric"

    valid_methods = ("best_nn_particle_mean", "best_particle", "last_nn_particle_mean")
    # @assert method in valid_methods "Invalid method: $method"

    my_valid_methods = ("best_particle_final", "mean_best_ensemble", "mean_final_ensemble",)
    valid_methods = (valid_methods..., my_valid_methods...)
    @assert (method in valid_methods || occursin("best_ensemble_me", method)) "Invalid method: $method"

    if method == "best_nn_particle_mean" 
        optimal = NCDataset(ds_path) do ds
            # mse_arr = ds.group["metrics"][string(metric, "_nn_mean")][:] # this is really the the particle with lowest MSE of those nearest to the ensemble mean, not nearest to the ensemble mean for the iteration with the lowest mean MSE
            mse_arr = ds.group["metrics"][string(metric, "_mean")][:] # I feel like this should just be _mean to get the iteration w/ the best mean mse
            replace!(mse_arr, NaN => Inf)
            mse_min_i = argmin(mse_arr)
            nn_mean_index = ds.group["metrics"]["nn_mean_index"][mse_min_i]
            u = ds.group["particle_diags"]["phi"][nn_mean_index, :, mse_min_i] # (n_particles, n_params, n_iterations)
            u_names = ds.group["ensemble_diags"]["param"][:]
            @info "Optimal parameters found at: iteration = $mse_min_i ; nn particle = $nn_mean_index"
            (; u_names, u)
        end
    elseif method == "last_nn_particle_mean"
        optimal = NCDataset(ds_path) do ds
            nn_mean_index = ds.group["metrics"]["nn_mean_index"][end - 1]
            u = ds.group["particle_diags"]["phi"][nn_mean_index, :, end - 1] # (n_particles, n_params, n_iterations)
            u_names = ds.group["ensemble_diags"]["param"][:]
            @info "Optimal parameters found at last iteration ; nn particle = $nn_mean_index"
            (; u_names, u)
        end
    elseif method == "best_particle"
        optimal = NCDataset(ds_path) do ds
            phi = ds.group["particle_diags"]["phi"][:] # (n_particles, n_params, n_iterations)
            part_min_i, iter_min_i = particle_diags_min_index(ds; metric = metric)
            u = phi[part_min_i, :, iter_min_i]
            u_names = ds.group["particle_diags"]["param"][:]
            @info "Optimal parameters found at: iteration = $iter_min_i ; particle = $part_min_i"
            (; u_names, u)
        end

    # my additions
    elseif method == "best_particle_final"
       optimal = NCDataset(ds_path) do ds
            phi = ds.group["particle_diags"]["phi"][:] # (n_particles, n_params, n_iterations)
            mse_mat = ds.group["particle_diags"][metric][:, end - 1] # only final iteration
            mse_min_i = argmin(mse_mat)
            u = phi[mse_min_i, :, end - 1]
            u_names = ds.group["particle_diags"]["param"][:]
            @info "Optimal parameters found at: iteration = $iter_min_i ; particle = $part_min_i"
            (; u_names, u)
        end

    elseif method == "mean_best_ensemble"
        optimal = NCDataset(ds_path) do ds
            mse_arr = ds.group["metrics"][string(metric, "_mean")][:] # I feel like this should just be _mean to get the iteration w/ the best mean mse
            replace!(mse_arr, NaN => Inf)
            mse_min_i = argmin(mse_arr)
            u = ds.group["particle_diags"]["phi"][:, :, mse_min_i] # (n_particles, n_params, n_iterations)
            u = mean(u, dims = 1)[:] # mean over all particles
            u_names = ds.group["ensemble_diags"]["param"][:]
            @info "Optimal parameters found at: iteration = $mse_min_i ; particle = $mse_min_i"
            (; u_names, u)
        end

    elseif method == "mean_final_ensemble"
        optimal = NCDataset(ds_path) do ds
            u = ds.group["particle_diags"]["phi"][:, :, end-1] # (n_particles, n_params, n_iterations)
            u = mean(u, dims = 1)[:] # mean over all particles
            u_names = ds.group["ensemble_diags"]["param"][:]
            @info "Optimal parameters found at: iteration = $mse_min_i ; nn particle = $nn_mean_index"
            (; u_names, u)
        end

    elseif occursin("best_ensemble_me", method) 
        # get the int out of best_ensemble_mean_{i}, need to figure out how to remove the brackets        
        i = parse(Int, strip( split(method, "_")[end], ['{', '}'])) # split by _ and take the last element, {#}, then strip the brackets and parse to Int
        mean_or_median = occursin("mean", method) ? mean : median 
        optimal = NCDataset(ds_path) do ds
            # calculate mse for ensemble mean profiles 
            sim_profiles = ds.group["particle_diags"]["g_full"][:] # (n_particles, n_obs, n_iterations) unscaled
            sim_profiles = mean_or_median(sim_profiles, dims = 1) # mean profile over all particles (should this be median?), so now is [1, n_obs, n_iterations]
            truth_profiles = ds.group["reference"]["y_full"][:] # (n_obs) unscaled
            truth_profiles = reshape(truth_profiles, (1, length(truth_profiles), 1))
            mse_arr = mean((sim_profiles .- truth_profiles).^2, dims = 2)[:] # (n_obs, 1)
            @info("mse sizing: ", size(mse_arr), size(sim_profiles), size(truth_profiles))
            mse_min_i = argmin(mse_arr)

            @info("sizing: ", size(ds.group["particle_diags"]["phi"]), i, mse_min_i )
            u = ds.group["particle_diags"]["phi"][i, :, mse_min_i] # (n_particles, n_params, n_iterations) - get particle i from iteration mean/median of profiles with the lowest MSE
            u_names = ds.group["ensemble_diags"]["param"][:]
            @info "Optimal parameters found at: iteration = $mse_min_i ; ensemble member = $i"
            (; u_names, u)
        end
    else
        error("Method $method not implemented yet.")
    end

    return optimal.u_names, optimal.u
end


"""
    optimal_mse(ds_path::String; method::String = "best_nn_particle_mean", metric::String = "mse_full")
Given path to Diagnostics.nc file, return minimum mse across iterations.

Inputs: 
 - `ds_path`: path to Diagnostics.nc file
 - `method`: method for computing minimum mse. Returns training and validation mse of:
        "best_particle" - particle with lowest mse in training (`metric` = "mse_full") or validation (`metric` = "mse_full_val") set.
        "best_nn_particle_mean" - particle nearest to ensemble mean for the iteration with lowest mse.
 - `metric` : mse metric to find the minimum of {"mse_full", "val_mse_full"}.

Returns: 
    - `mse_train`: mse for training set
    - `mse_val`: mse for validation set
"""

function optimal_mse(ds_path::String; method::String = "best_nn_particle_mean", metric::String = "mse_full")

    valid_metrics = ("mse_full", "val_mse_full")
    @assert metric in valid_metrics "Invalid metric: $metric"

    valid_methods = ("best_nn_particle_mean", "best_particle", "last_nn_particle_mean")
    @assert method in valid_methods "Invalid method: $method"

    if method == "best_nn_particle_mean"
        optimal = NCDataset(ds_path) do ds
            mse_arr = ds.group["metrics"][string(metric, "_nn_mean")][:]
            replace!(mse_arr, NaN => Inf)
            mse_min_i = argmin(mse_arr)
            mse_train = ds.group["metrics"]["mse_full_nn_mean"][mse_min_i]
            # if val dataset provided, return val mse
            if ("val_mse_full_nn_mean" in keys(ds.group["metrics"]))
                mse_val = ds.group["metrics"]["val_mse_full_nn_mean"][:][mse_min_i]
            else
                mse_val = nothing
            end
            @info "Optimal mse found at: iteration = $mse_min_i"
            (; mse_train, mse_val)
        end
    elseif method == "last_nn_particle_mean"
        optimal = NCDataset(ds_path) do ds
            mse_train = ds.group["metrics"]["mse_full_nn_mean"][end - 1]
            # if val dataset provided, return val mse
            if ("val_mse_full_nn_mean" in keys(ds.group["metrics"]))
                mse_val = ds.group["metrics"]["val_mse_full_nn_mean"][:][end - 1]
            else
                mse_val = nothing
            end
            @info "Optimal mse found at last iteration."
            (; mse_train, mse_val)
        end
    elseif method == "best_particle"
        optimal = NCDataset(ds_path) do ds
            part_min_i, iter_min_i = particle_diags_min_index(ds; metric = metric)
            mse_mat = ds.group["particle_diags"]["mse_full"][:] # (n_particles, n_iterations)
            mse_train = mse_mat[part_min_i, iter_min_i]
            # if val dataset provided, return val mse
            if ("val_mse_full" in keys(ds.group["particle_diags"]))
                mse_mat_val = ds.group["particle_diags"]["val_mse_full"][:]
                mse_val = mse_mat_val[part_min_i, iter_min_i]
            else
                mse_val = nothing
            end
            @info "Optimal mse found at: iteration = $iter_min_i ; particle = $part_min_i"
            (; mse_train, mse_val)
        end
    end

    return optimal.mse_train, optimal.mse_val
end

"""
    Given path to Diagnostics.nc file, return mean(mse across particles) in the final iteration for training and validation set.
"""
function final_mse(ds_path::String; metric::String = "mse_full_mean")
    final = NCDataset(ds_path) do ds
        mse_mean = ds.group["metrics"][metric][:]
        mse_train = mse_mean[length(mse_mean) - 1]
        val_metric = string("val_", metric)
        # if val dataset provided, return val mse
        if (val_metric in keys(ds.group["metrics"]))
            ds_mse_val_mean = ds.group["metrics"][val_metric][:]
            mse_val = ds_mse_val_mean[length(ds_mse_val_mean) - 1]
        else
            mse_val = nothing
        end
        (; mse_train, mse_val)
    end

    return final.mse_train, final.mse_val
end

"""
    Given path to Diagnostics.nc file, return parameter names and mean values in final iteration.
"""
function final_parameters(ds_path::String)
    final = NCDataset(ds_path) do ds
        ds_phi_mean = ds.group["ensemble_diags"]["phi_mean"]
        u = ds_phi_mean[:, size(ds_phi_mean, 2) - 1]
        u_names = ds.group["ensemble_diags"]["param"][:]
        (; u_names, u)
    end
    return final.u_names, final.u
end
