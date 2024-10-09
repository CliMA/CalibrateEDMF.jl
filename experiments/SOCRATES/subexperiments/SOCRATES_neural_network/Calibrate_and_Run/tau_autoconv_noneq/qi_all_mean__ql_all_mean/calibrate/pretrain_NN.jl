# pretrain the neural network

# NOTE: To do -- switch to using closure functions...
# include("/home/jbenjami/Research_Schneider/CliMa/TurbulenceConvection.jl/src/closures/neural_microphysics_relaxation_timescales.jl") # see definitions there...

using Flux
using JLD2
using ProgressMeter: @showprogress
using Statistics

thisdir = dirname(@__FILE__)

FT = Float32
ρ_0, T_0, q_liq_0, q_ice_0, w_0 = FT(1.), FT(273.15), FT(1e-4), FT(1e-5), FT(1e-4) # characteristic values
x_0_characteristic = [ρ_0, T_0, q_liq_0, q_ice_0, w_0] 


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

"""
# Would it be better to use Temperature in Celsius?
# What about q? Could those be log scale? that wouldn't work bc 0 would be -inf... but how can we reduce the range? We could do log(q + 1e-10) or something like that...
# The same w/ w, though maybe linear is fine there and it doesn't matter as much

"""

function predict_τ(ρ,T,q, w, NN; FT=Float32, norm = x_0_characteristic)
    # normalize
    x_0 = FT.([ρ, T, q.liq, q.ice, w]) ./ norm
    log_τ_liq, log_τ_ice = NN(x_0)
    return exp10(log_τ_liq), exp10(log_τ_ice)
end

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
# data = [(x_0_characteristic ./ x_0_characteristic, truth)]


# many samples (to avoid overfitting)
n = 250
ρ_0 = FT.(1. .-  (rand(n).-.1))
T_0 = FT.((273.15) .+ 100 .* (rand(n).-0.5))
q_liq_0 = FT.(rand(n) / 1e3)
q_ice_0 = FT.(rand(n) / 1e4)

w_0 = FT.(rand(n)/1e2) # max at 1e-2 (1 cm/s)
# τ_liq_true = FT.(maximum(q_liq_0) ./ q_liq_0 * 1e1 .+ rand(n).*1e1) # random, fast for large q_liq
# τ_ice_true = FT.(T_0 ./ minimum(T_0) .*  maximum(q_ice_0) ./ q_ice_0  * 1e5 .+ rand(n).*1e5 ) # random, fast for large q_ice, slow for high T

q_con_0 = 10 .^ -((rand(n)) * 6 .+ 2) # 6 orders of magniude, maxing at 1e-2
q_con_0_log = log10.(q_con_0)
τ_liq_true = FT.(maximum(q_con_0) ./ q_con_0) |> x-> x + (x./2).*rand(n)  # random, fast for large q_li
τ_ice_true = FT.( ((T_0 .- minimum(T_0)) ./ ( maximum(T_0) - minimum(T_0))).^1 ) .+  ((maximum(q_con_0_log).-minimum(q_con_0_log)) ./ (maximum(q_con_0_log).-q_con_0_log)  ).^-1  |> x-> x + (x./2).*rand(n)  # fast for either high q or low T. data is scaled 0 to 1 and then scaled back out afterwards. # we didnt add an offset so this wil be 0 if the min T and q overlap in index but that's very unlikely (and maybe wouldnt hurt training too much? idk...)
τ_ice_true = τ_ice_true .* 2 # scale from 0 to 3 to 0 to 6 (noise scaled up from 0 ->(1+1=2) to to 0->(2+ 2/2 = 3)
τ_ice_true = 10 .^ τ_ice_true # scale from 0 to 6 to 1e0 to 1e6


# Plot data aif you have UnicodePlots installed
# UnicodePlots.scatterplot(q_con_0, τ_liq_true, yscale=:log10,xscale=:log10, height=20, width=50), println("dd"), UnicodePlots.scatterplot(T_0, τ_ice_true,  yscale=:log10, height=20, width=50), UnicodePlots.scatterplot(q_con_0, τ_ice_true,  yscale=:log10, xscale=:log10, height=20, width=50)

truth = hcat(τ_liq_true, τ_ice_true)
truth = log10.(truth)
input = hcat(ρ_0, T_0, q_liq_0, q_ice_0, w_0) ./ x_0_characteristic' # would be nice to have logs for liq, ice, w but those would be -inf at 0...
data = (input, truth)
data = tuple.(eachrow(data[1]), eachrow(data[2])) # make it a tuple of tuples # I think it's supposed to be a list of [(x1,y1), (x2,y2),...] 
# penalty(model) = sum(x->sum(abs2, x), Flux.params(model)) # L2 penalty (very slow...)

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


speed = 1e-5
opt = Descent(speed)
loss_func(func) = (model,x, y) -> func(model(x), y) + penalty(model)
# loss_func(func) = (model,x, y) -> func(model(x), log10.(y))
@showprogress for epoch in 1:round(Int,1.5/speed)
    Flux.train!( loss_func(Flux.Losses.mse), NN, data, opt)
end
@info("INFO", truth, NN(x_0_characteristic ./x_0_characteristic))

# info
prediction = NN(input')
@info("model vs. truth", prediction, truth')
@info("stats", cor( prediction', truth''),loss_func(Flux.Losses.mse)(NN, input', truth') )

# save to disk...
savepath = "pretrained_NN.jld2"
@info("INFO", "saving to $savepath")
params, repr = Flux.destructure(NN)
JLD2.save( joinpath(thisdir, savepath), Dict("params"=>params, "re"=>repr, "x_0_characteristic" => x_0_characteristic))

# flux train and save output
# p1 = UnicodePlots.scatterplot(truth'[1,:], prediction[1,:],); p2 = UnicodePlots.scatterplot(truth'[2,:], prediction[2,:],); UnicodePlots.lineplot!(p1, 1:n, 1:n), UnicodePlots.lineplot!(p2, 1:n,  1:n) # if you use unicode plots, you can vizualize the correlation between the truth and the prediction




