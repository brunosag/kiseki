module LEEA

import Lux
using Random: Xoshiro
using Polyester: @batch
using ..Data: load_MNIST
using ..Models: CNN_2C2D_MNIST

export train_LEEA

const logitcrossentropy = Lux.CrossEntropyLoss(; logits = Val(true))

@kwdef struct LEEAConfig
    P::Int = 1000       # population size
    r::Float32 = 0.04   # mutation rate
    m₀::Float32 = 0.03  # initial mutation power
    γ::Float32 = 0.99   # mutation power decay
    pₛ::Float32 = 0.4   # selection proportion
    s::Float32 = 0.5    # sexual reproduction proportion
    d::Float32 = 0.2    # fitness inheritance decay
end

function train_LEEA(; seed::Int, batchsize::Int, generations::Int)
    rng = Xoshiro(seed)
    (; P, r, m₀, γ, pₛ, s, d) = LEEAConfig()

    train_dataloader, _ = load_MNIST(; rng, batchsize, balanced = true)
    model = CNN_2C2D_MNIST
    _, st = Lux.setup(rng, model)

    pop = stack(Vector(Lux.initialparameters(rng, model).params) for _ in 1:P)
    f = zeros(Float32, P)

    for i in 1:generations
        X, Y = popfirst!(train_dataloader)

        @batch for j in 1:P
            θⱼ = NamedTuple{(:params,)}((@view(pop[:, j]),))
            Ŷ, _ = model(X, θⱼ, st)
            f[j] = 1 / (1 + logitcrossentropy(Ŷ, Y))
        end

        wheel = partialsort(f, 1:round(Int, pₛ * P), rev = true)
        @show wheel
    end

    return
end

end
