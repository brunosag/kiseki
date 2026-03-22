@kwdef struct SGD <: AbstractOptimizer
    η::Float32 = 0.01f0   # learning rate
end

mutable struct SGDState{P, S} <: AbstractOptimizerState
    θ::P
    st::S
end

function init(opt::SGD, model, dev, rng)
    θ, _ = Lux.setup(rng, model) |> dev
    st = Optimisers.setup(Descent(opt.η), θ)

    return SGDState(θ, st)
end

function step!(opt::SGD, ops, ws, model, st, X, Y, rng)
    loss, pb = Zygote.pullback(ops.θ) do θ
        Ŷ, _ = model(X, θ, st)
        return logitcrossentropy(Ŷ, Y)
    end
    ∇θ = pb(1.0f0)[1]

    ops.st, ops.θ = Optimisers.update!(ops.st, ops.θ, ∇θ)

    return loss
end

get_best_params(ops::SGDState) = ops.θ
