# pretrain the neural network

# NOTE: To do -- switch to using closure functions...

using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl")
using Flux # should we actually add this as a dependency? I think we should not bc it's so rarely used... makes it annoying to train tho lmao...
using JLD2
using ProgressMeter: @showprogress
using Statistics
using NCDatasets

using StatsBase

thisdir = dirname(@__FILE__)

# ========================================================================================================================= #

# FT = Float32
# ρ_0, T_0, q_liq_0, q_ice_0, w_0 = FT(1.), FT(273.15), FT(1e-4), FT(1e-7), FT(1e-3) # characteristic values
# x_0_characteristic = [ρ_0, T_0, q_liq_0, q_ice_0, w_0] 


# # Copied code from Dense layer and simply renamed it (was too untenable to train easily)
# struct Exponential{F<:Function}
#     σ::F
# end
# Exponential() = Exponential(exp10)
# Flux.@functor Exponential # turns it into something that can return params and it's construction method

# function (a::Exponential)(x::AbstractArray)
#     σ = a.σ
#     return σ.(x)
# end

# """
# # Would it be better to use Temperature in Celsius?
# # What about q? Could those be log scale? that wouldn't work bc 0 would be -inf... but how can we reduce the range? We could do log(q + 1e-10) or something like that...
# # The same w/ w, though maybe linear is fine there and it doesn't matter as much

# # ensure matches formula in /home/jbenjami/Research_Schneider/CliMa/TurbulenceConvection.jl/src/closures/neural_microphysics_relaxation_timescales.jl

# """

# function predict_τ(ρ,T,q, w, NN; FT=Float32, norm = x_0_characteristic)
#     # normalize
#     x_0 = FT.([ρ, T, q.liq, q.ice, w]) ./ norm
#     log_τ_liq, log_τ_ice = NN(x_0)
#     return exp10(log_τ_liq), exp10(log_τ_ice)
# end

include("/home/jbenjami/Research_Schneider/CliMa/TurbulenceConvection.jl/src/closures/neural_microphysics_relaxation_timescales.jl") # see definitions there...

# ========================================================================================================================= #

function τ_neural_network(L) # this could actually go anywhere, including in calibrate edmf since we just need to pass the repr location...
    # pretrain towards the correct τ or something (default or maybe the simple one.)
    NN = Flux.Chain(
        Flux.Dense(L  => 10, Flux.relu, bias=true), # relu requires very short timesteps in Descent
        Flux.Dense(10 =>  8, Flux.relu, bias=true),
        Flux.Dense( 8 =>  4, Flux.relu, bias=true),
        Flux.Dense( 4 =>  2, bias=false),
        Flux.Dense( 2 =>  2, bias=true), # no activation, allow negative outputs... it should work in log space...

        # Flux.Dense( 8 =>  8, Flux.tanh, bias=true),
        # Flux.Dense( 8 =>  2, bias=true), # no activation, allow negative outputs... it should work in log space...
        # Exponential() # hard to calibrate with
        )
    return NN
end


# # single sample
# τ_liq_true, τ_ice_true = FT(1e1), FT(1e5)
# penalty(model) = 0.
# truth = [τ_liq_true, τ_ice_true]
# data = [(x_0_characteristic ./ x


use_LES_inferred_data = true

if use_LES_inferred_data
    # LES_inferred_datafile = "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Reference/Output_Inferred_Data/τ_inferred_RFAll_combined_vector.nc"
    # LES_inferred_data = NCDatasets.Dataset(LES_inferred_datafile, "r")

    # T     = nomissing(LES_inferred_data["temperature_mean"][:], NaN)
    # q_liq = nomissing(LES_inferred_data["ql_mean"][:], NaN)
    # q_ice = nomissing(LES_inferred_data["qi_mean"][:], NaN)
    # τ_liq = nomissing(LES_inferred_data["τ_cond_evap"][:], NaN) # this is broken bc PCC (cond/evap) seems to be broken in Atlas output files...
    # τ_ice = nomissing(LES_inferred_data["τ_sub_dep"][:], NaN)
    # p     = nomissing(LES_inferred_data["p_mean"][:], NaN)
    # ρ     = nomissing(LES_inferred_data["ρ_mean"][:], NaN)
    # # Ni    = nomissing(LES_inferred_data["ni_mean"][:], NaN)
    # w = zeros(FT, size(T)) .+ ( (rand(FT,length(T)) .- FT(0.5)) .* FT(1e-2) ) # 0 plus a little jitter (we don't have w in the LES data since it'e the entire mean area

    LES_inferred_datafile = "/home/jbenjami/Research_Schneider/CliMa/CalibrateEDMF.jl/experiments/SOCRATES/Reference/Output_Inferred_Data/SOCRATES_Atlas_LES_inferred_timescales.nc"
    LES_inferred_data = NCDatasets.Dataset(LES_inferred_datafile, "r")

    T     = vec(nomissing(LES_inferred_data["T"][:], NaN))
    q_liq = vec(nomissing(LES_inferred_data["q_liq"][:], NaN))
    q_ice = vec(nomissing(LES_inferred_data["q_ice"][:], NaN))
    τ_liq = vec(nomissing(LES_inferred_data["τ_cond_evap"][:], NaN)) # this is broken bc PCC (cond/evap) seems to be broken in Atlas output files...
    τ_ice = vec(nomissing(LES_inferred_data["τ_sub_dep"][:], NaN))
    p     = vec(nomissing(LES_inferred_data["p"][:], NaN))
    ρ     = vec(nomissing(LES_inferred_data["ρ"][:], NaN))
    # Ni    = nomissing(LES_inferred_data["ni_mean"][:], NaN)
    w = zeros(FT, size(T)) .+ ( (rand(FT,length(T)) .- FT(0.5)) .* FT(1e-2) ) # 0 plus a little jitter (we don't have w in the LES data since it'e the entire mean area


    valid = (isfinite.(τ_liq) .& isfinite.(τ_ice)) # should this be some kind of threshold?

    T = T[valid]
    q_liq = q_liq[valid]
    q_ice = q_ice[valid]
    τ_liq = τ_liq[valid]
    τ_ice = τ_ice[valid]
    p = p[valid]
    ρ = ρ[valid]
    # Ni = Ni[valid]
    w = w[valid]

    # We have too much data and a lot of it is similar, so we can subsample it to make training faster
    N_subset = Int(1e4)
    random_indices = StatsBase.sample(1:length(T), N_subset, replace=false)

    T = T[random_indices]
    q_liq = q_liq[random_indices]
    q_ice = q_ice[random_indices]
    τ_liq = τ_liq[random_indices]
    τ_ice = τ_ice[random_indices]
    p = p[random_indices]
    ρ = ρ[random_indices]
    # Ni = Ni[random_indices]
    w = w[random_indices]

else

    # many samples (to avoid overfitting)
    n = 250
    ρ = FT.(1. .-  (rand(n).-.1))
    T = FT.((273.15) .+ 100 .* (rand(n).-0.5))
    q_liq = FT.(rand(n) / 1e3)
    q_ice = FT.(rand(n) / 1e4)

    w = FT.(rand(n)/1e2) # max at 1e-2 (1 cm/s)
    # τ_liq = FT.(maximum(q_liq) ./ q_liq * 1e1 .+ rand(n).*1e1) # random, fast for large q_liq
    # τ_ice = FT.(T ./ minimum(T) .*  maximum(q_ice) ./ q_ice  * 1e5 .+ rand(n).*1e5 ) # random, fast for large q_ice, slow for high T

    q_con_0 = 10 .^ -((rand(n)) * 6 .+ 2) # 6 orders of magniude, maxing at 1e-2
    q_con_0_log = log10.(q_con_0)
    τ_liq = FT.(maximum(q_con_0) ./ q_con_0) |> x-> x + (x./2).*rand(n)  # random, fast for large q_li
    τ_ice = FT.( ((T .- minimum(T)) ./ ( maximum(T) - minimum(T))).^1 ) .+  ((maximum(q_con_0_log).-minimum(q_con_0_log)) ./ (maximum(q_con_0_log).-q_con_0_log)  ).^-1  |> x-> x + (x./2).*rand(n)  # fast for either high q or low T. data is scaled 0 to 1 and then scaled back out afterwards. # we didnt add an offset so this wil be 0 if the min T and q overlap in index but that's very unlikely (and maybe wouldnt hurt training too much? idk...)
    τ_ice = τ_ice .* 2 # scale from 0 to 3 to 0 to 6 (noise scaled up from 0 ->(1+1=2) to to 0->(2+ 2/2 = 3)
    τ_ice = 10 .^ τ_ice # scale from 0 to 6 to 1e0 to 1e6


    # Plot data aif you have UnicodePlots installed
    # UnicodePlots.scatterplot(q_con_0, τ_liq, yscale=:log10,xscale=:log10, height=20, width=50), println("dd"), UnicodePlots.scatterplot(T, τ_ice,  yscale=:log10, height=20, width=50), UnicodePlots.scatterplot(q_con_0, τ_ice,  yscale=:log10, xscale=:log10, height=20, width=50)


end

truth = log10.(hcat(τ_liq, τ_ice))
# input = hcat(ρ, T, q_liq, q_ice, w) ./ x_0_characteristic' # would be nice to have logs for liq, ice, w but those would be -inf at 0...
ρ, T, q_liq, q_ice, w = prepare_for_NN(ρ, T, q_liq, q_ice, w)
input = hcat(ρ, T, q_liq, q_ice, w) # would be nice to have logs for liq, ice, w but those would be -inf at 0...


data = (input, truth)
data = tuple.(eachrow(data[1]), eachrow(data[2])) # make it a tuple of tuples # I think it's supposed to be a list of [(x1,y1), (x2,y2),...] 
# penalty(model) = sum(x->sum(abs2, x), Flux.params(model)) # L2 penalty (very slow...)


retrain = true

NN = τ_neural_network(length(x_0_characteristic))
# λ=sum(length, Flux.params(NN)) ./ Flux.Losses.mse(NN(input'), truth') # 1/loss is a good guess for the scale of the loss
# function penalty(model::Chain; λ=λ) # can't call param inside loss fcn, too slow
#     penalty = FT(0. )
#     for layer in model.layers
#         penalty += sum(abs2, layer.weight)
#         penalty += sum(abs2, layer.bias)
#     end
#     return sum(penalty) / λ
# end
    # sum( [sum(abs2, x) for x in Flux.params(model)] ) # L2 penalty (very slow...)
penalty(model) = FT(0.) # no penalty, hope the noise and small model preclude overfitting...e


# speed = 1e-5
speed = 1e-3
opt = Descent(speed)
loss_func(func) = (model,x, y) -> func(model(x), y) + penalty(model)
# loss_func(func) = (model,x, y) -> func(model(x), log10.(y))

if retrain
    @showprogress for epoch in 1:round(Int,1.5/speed)
        Flux.train!( loss_func(Flux.Losses.mse), NN, data, opt)
    end
else
    # load pretrained model
    nn_path = joinpath(thisdir, "pretrained_NN.jld2")
    nn_pretrained_params, nn_pretrained_repr, nn_pretrained_x_0_characteristic = JLD2.load(nn_path, "params", "re", "x_0_characteristic")
    NN = vec_to_NN(nn_pretrained_params, nn_pretrained_repr)
    # x_0_characteristic = nn_pretrained_x_0_characteristic
end


# info
prediction = NN(input')
@info("model vs. truth", prediction, truth')
@info("stats", cor( prediction', truth''), loss_func(Flux.Losses.mse)(NN, input', truth') )

# save to disk...
if retrain
    savepath = "pretrained_NN.jld2"
    @info("INFO", "saving to $savepath")
    params, repr = Flux.destructure(NN)
    JLD2.save( joinpath(thisdir, savepath), Dict("params"=>params, "re"=>repr, "x_0_characteristic" => x_0_characteristic))
end
# flux train and save output
# p1 = UnicodePlots.scatterplot(truth'[1,:], prediction[1,:],); p2 = UnicodePlots.scatterplot(truth'[2,:], prediction[2,:],); UnicodePlots.lineplot!(p1, 1:n, 1:n), UnicodePlots.lineplot!(p2, 1:n,  1:n) # if you use unicode plots, you can vizualize the correlation between the truth and the prediction




