@kwdef struct LEEA <: AbstractOptimizer
    N::Int = 1000        # population size
    r::Float32 = 0.04    # mutation rate
    m₀::Float32 = 0.03   # initial mutation power
    γ::Float32 = 0.99    # mutation power decay
    p::Float32 = 0.4     # selection proportion
    s::Float32 = 0.5     # sexual reproduction proportion
    d::Float32 = 0.2     # fitness inheritance decay
    pat_lim::Int = 50    # generations to wait before decaying m
end

mutable struct LEEAState{M <: AbstractMatrix{Float32}} <: AbstractOptimizerState
    re::Restructure
    P::M
    O::M
    fₚ::Vector{Float32}
    fₒ::Vector{Float32}
    pₐ::Vector{Int}
    p₁::Vector{Int}
    p₂::Vector{Int}
    m::Float32
    pat::Int
    is_first_step::Bool
end

function init(opt::LEEA, model, dev, rng)
    _, re = destructure(Lux.initialparameters(TaskLocalRNG(), model))

    P = stack([destructure(Lux.initialparameters(rng, model))[1] for _ in 1:opt.N]) |> dev
    O = similar(P)

    Nₛ = round(Int, opt.s * opt.N)
    Nₐ = opt.N - Nₛ

    alloc(T′, n) = Vector{T′}(undef, n)
    fₚ, fₒ = alloc(Float32, opt.N), alloc(Float32, opt.N)
    pₐ, p₁, p₂ = alloc(Int, Nₐ), alloc(Int, Nₛ), alloc(Int, Nₛ)

    return LEEAState(re, P, O, fₚ, fₒ, pₐ, p₁, p₂, opt.m₀, 0, true)
end

function evaluate_fitness!(opt::LEEA, ops::LEEAState, model, st, X, Y)
    best_loss = Inf32

    for j in 1:opt.N
        θ = ops.re(@view ops.P[:, j])
        Ŷ, _ = model(X, θ, st)
        L = Float32(logitcrossentropy(Ŷ, Y))

        if L < best_loss
            best_loss = L
        end

        ops.fₒ[j] = 1.0f0 / (1.0f0 + L)
    end

    return best_loss
end

function inherit_fitness!(opt::LEEA, ops::LEEAState)
    d′ = 1.0f0 - opt.d
    half_d′ = 0.5f0 * d′
    Nₐ = length(ops.pₐ)

    for j in 1:opt.N
        if ops.is_first_step
            continue
        elseif j <= Nₐ
            ops.fₒ[j] += ops.fₚ[ops.pₐ[j]] * d′
        else
            ops.fₒ[j] += (ops.fₚ[ops.p₁[j - Nₐ]] + ops.fₚ[ops.p₂[j - Nₐ]]) * half_d′
        end
    end
    return
end

function select_parents!(opt::LEEA, ops::LEEAState, rng)
    wheel = partialsortperm(ops.fₒ, 1:round(Int, opt.p * opt.N), rev = true)
    weights = Weights(@view ops.fₒ[wheel])

    sample!(rng, wheel, weights, ops.pₐ)
    sample!(rng, wheel, weights, ops.p₁)
    sample!(rng, wheel, weights, ops.p₂)

    collisions = ops.p₁ .== ops.p₂

    while any(collisions)
        n_collisions = sum(collisions)
        ops.p₂[collisions] .= sample(rng, wheel, weights, n_collisions)
        collisions = ops.p₁ .== ops.p₂
    end
    return
end

function reproduce_assexual!(opt::LEEA, ops::LEEAState, rng)
    θ_len = size(ops.P, 1)
    Nₐ = length(ops.pₐ)

    pₐ = similar(ops.P, Int, Nₐ)
    copyto!(pₐ, ops.pₐ)

    u₁ = similar(ops.P, Float32, θ_len, Nₐ)
    u₂ = similar(ops.P, Float32, θ_len, Nₐ)
    rand!(rng, u₁)
    rand!(rng, u₂)

    @views ops.O[:, 1:Nₐ] .= ops.P[:, pₐ] .+ (u₁ .< opt.r) .* ops.m .* (2.0f0 .* u₂ .- 1.0f0)
    return
end

function reproduce_sexual!(ops::LEEAState, rng)
    θ_len = size(ops.P, 1)
    Nₐ = length(ops.pₐ)
    Nₛ = length(ops.p₁)

    p₁ = similar(ops.P, Int, Nₛ)
    p₂ = similar(ops.P, Int, Nₛ)
    copyto!(p₁, ops.p₁)
    copyto!(p₂, ops.p₂)

    u = similar(ops.P, Float32, θ_len, Nₛ)
    rand!(rng, u)

    @views ops.O[:, (Nₐ + 1):end] .= ifelse.(u .< 0.5f0, ops.P[:, p₁], ops.P[:, p₂])
    return
end

function step!(opt::LEEA, ops::LEEAState, model, st, X, Y, rng)
    best_loss = evaluate_fitness!(opt, ops, model, st, X, Y)
    inherit_fitness!(opt, ops)
    select_parents!(opt, ops, rng)
    reproduce_assexual!(opt, ops, rng)
    reproduce_sexual!(ops, rng)

    ops.P, ops.O = ops.O, ops.P
    ops.fₚ, ops.fₒ = ops.fₒ, ops.fₚ

    return best_loss
end

function get_best_params(ops::LEEAState)
    best_idx = argmax(ops.fₚ)
    return ops.re(@view ops.P[:, best_idx])
end

function update_scheduler!(opt::LEEA, ops::LEEAState, acc, best_acc)
    if acc > best_acc
        ops.pat = 0
    else
        ops.pat += 1
    end

    if ops.pat >= opt.pat_lim
        ops.m *= opt.γ
        ops.pat = 0
    end
    return
end
