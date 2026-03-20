@kwdef struct SGD <: AbstractOptimizer
    η::Float32 = 0.01f0   # learning rate
end

mutable struct SGDState{P, S} <: AbstractOptimizerState
    θ::P
    ts::S
end

function init(opt::SGD, model, dev, rng)
    θ, st = Lux.setup(rng, model) |> dev
    ts = Lux.Training.TrainState(model, θ, st, Descent(opt.η))

    return SGDState(θ, ts)
end

function step!(opt::SGD, ops, ws, model, st, X, Y, rng)
    _, loss, _, ops.ts = Lux.Training.single_train_step!(
        AutoZygote(), logitcrossentropy, (X, Y), ops.ts
    )
    return loss
end

get_best_params(ops::SGDState) = ops.θ
