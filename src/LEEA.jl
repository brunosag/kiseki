module LEEA

import Lux
using Random: Xoshiro, rand!
using Polyester: @batch
using StatsBase: Weights, sample, sample!
using Base.Threads: nthreads, threadid
using ..Data: load_MNIST
using ..Models: CNN_2C2D_MNIST

export train_LEEA

const logitcrossentropy = Lux.CrossEntropyLoss(; logits = Val(true))

@kwdef struct LEEAConfig
    N::Int = 1000       # population size
    r::Float32 = 0.04   # mutation rate
    m::Float32 = 0.03   # mutation power
    γₘ::Float32 = 0.99  # mutation power decay
    p::Float32 = 0.4    # selection proportion
    s::Float32 = 0.5    # sexual reproduction proportion
    γᵢ::Float32 = 0.2   # fitness inheritance decay
end

function evaluate_population!(f, model, P, X, Y, st, N)
    return @batch for j in 1:N
        θⱼ = NamedTuple{(:params,)}((@view(P[:, j]),))
        Ŷ, _ = model(X, θⱼ, st)
        f[j] = 1.0f0 / (1.0f0 + logitcrossentropy(Ŷ, Y))
    end
end

function select_parents!(f, pₐ, p₁, p₂, rng, N, p)
    wheel = partialsortperm(f, 1:round(Int, p * N), rev = true)
    weights = Weights(@view f[wheel])

    sample!(rng, wheel, weights, pₐ)
    sample!(rng, wheel, weights, p₁)
    sample!(rng, wheel, weights, p₂)

    collisions = p₁ .== p₂
    while any(collisions)
        n_collisions = sum(collisions)
        p₂[collisions] .= sample(rng, wheel, weights, n_collisions)
        collisions = p₁ .== p₂
    end
    return
end

function reproduce_assexual!(O, P, pₐ, U₁, U₂, rngs, Nₐ, r, m)
    return @batch for j in 1:Nₐ
        tid = threadid()
        u₁ = U₁[tid]
        u₂ = U₂[tid]
        pⱼ = pₐ[j]

        rand!(rngs[tid], u₁)
        rand!(rngs[tid], u₂)

        @. O[:, j] = P[:, pⱼ] + (u₁ < r) * m * (2.0f0 * u₂ - 1.0f0)
    end
end

function reproduce_sexual!(O, P, p₁, p₂, U, rngs, Nₛ, Nₐ)
    return @batch for j in 1:Nₛ
        tid = threadid()
        u = U[tid]
        p₁ⱼ = p₁[j]
        p₂ⱼ = p₂[j]

        rand!(rngs[tid], u)

        @. O[:, Nₐ + j] = ifelse(u < 0.5f0, P[:, p₁ⱼ], P[:, p₂ⱼ])
    end
end

function train_LEEA(; seed::Int, batchsize::Int, generations::Int)
    (; N, r, m, γₘ, p, s, γᵢ) = LEEAConfig()

    n_threads = nthreads()
    rng = Xoshiro(seed)
    rngs = [Xoshiro(seed + t) for t in 1:n_threads]

    train_dataloader, _ = load_MNIST(; rng, batchsize, balanced = true)
    model = CNN_2C2D_MNIST

    _, st = Lux.setup(rng, model)
    st = Lux.testmode(st)
    θ_len = Lux.parameterlength(model)

    P = stack(Vector(Lux.initialparameters(rng, model).params) for _ in 1:N)
    O = similar(P)

    Nₛ = round(Int, s * N)
    Nₐ = N - Nₛ

    alloc(T, n) = Vector{T}(undef, n)
    f = alloc(Float32, N)
    pₐ, p₁, p₂ = alloc(Int, Nₐ), alloc(Int, Nₛ), alloc(Int, Nₛ)
    Uₛ, U₁, U₂ = ntuple(_ -> [alloc(Float32, θ_len) for _ in 1:n_threads], 3)

    for i in 1:generations
        X, Y = popfirst!(train_dataloader)

        evaluate_population!(f, model, P, X, Y, st, N)
        select_parents!(f, pₐ, p₁, p₂, rng, N, p)
        reproduce_assexual!(O, P, pₐ, U₁, U₂, rngs, Nₐ, r, m)
        reproduce_sexual!(O, P, p₁, p₂, Uₛ, rngs, Nₛ, Nₐ)

        m *= γₘ
        P, O = O, P
    end
    return
end

end
