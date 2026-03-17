module experiment

import Lux
import Base: run
using Printf
using Random: AbstractRNG, Xoshiro, TaskLocalRNG
using Optimisers: destructure
import ..optimizers: init
using ..optimizers
using ..models
using ..data

export Experiment, ExperimentState, init, run

@kwdef struct Experiment
    seed::Int = 42
    batchsize::Int = 500
    max_i::Int = 5
    target_acc::Float64 = 1.0
    opt::AbstractOptimizer = LEEA()
end

mutable struct ExperimentState
    rng
    ops
    train_dataloader
    test_dataloader
    best_acc
    pat
    i
end

function init(exp::Experiment, dev, model)
    rng = Xoshiro(exp.seed)
    ops = init(exp.opt, model, rng, dev)
    train_dataloader, test_dataloader = load_MNIST(rng, exp.batchsize)
    best_acc = 0.0
    pat = 0
    i = 1

    return ExperimentState(
        rng, ops, train_dataloader, test_dataloader, best_acc, pat, i
    )
end

function run(exp::Experiment; est::Union{ExperimentState, Nothing} = nothing)
    dev = Lux.gpu_device()
    model = CNN_2C2D_MNIST
    st = Lux.testmode(Lux.initialstates(TaskLocalRNG(), model))
    re = destructure(Lux.initialparameters(TaskLocalRNG(), model))[2]

    if isnothing(est)
        est = init(exp, dev, model)
    end

    while est.i <= exp.max_i && est.best_acc < exp.target_acc
        t₀ = time()
        X, Y = popfirst!(est.train_dataloader) |> dev

        L = step!(exp.opt, est.ops, re, model, st, X, Y, est.rng)

        Δt = time() - t₀
        @printf "i = %-*d    Δt = %.2fs    L = %.4f\n" ndigits(exp.max_i) est.i Δt L

        est.i += 1
    end
    return
end

end
