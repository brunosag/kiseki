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

mutable struct LEEAState{M <: AbstractMatrix{Float32}, R} <: AbstractOptimizerState
    re::R
    P::M
    fₚ::Vector{Float32}
    m::Float32
    pat::Int
    is_first_step::Bool
end

mutable struct LEEAWorkspace{M <: AbstractMatrix{Float32}}
    O::M
    fₒ::Vector{Float32}
    pₐ::Vector{Int}
    p₁::Vector{Int}
    p₂::Vector{Int}
end

function init(opt::LEEA, model, dev, rng)
    _, re = destructure(Lux.initialparameters(TaskLocalRNG(), model))
    P = stack([destructure(Lux.initialparameters(rng, model))[1] for _ in 1:opt.N]) |> dev
    fₚ = Vector{Float32}(undef, opt.N)

    return LEEAState(re, P, fₚ, opt.m₀, 0, true)
end

function init_workspace(opt::LEEA, ops)
    O = similar(ops.P)
    Nₛ = round(Int, opt.s * opt.N)
    Nₐ = opt.N - Nₛ

    alloc(T′, n) = Vector{T′}(undef, n)
    fₒ = alloc(Float32, opt.N)
    pₐ, p₁, p₂ = alloc(Int, Nₐ), alloc(Int, Nₛ), alloc(Int, Nₛ)

    return LEEAWorkspace(O, fₒ, pₐ, p₁, p₂)
end

@inline function evaluate_individual!(ops, ws, model, st, X, Y, j)
    θ = ops.re(@view ops.P[:, j])
    Ŷ, _ = model(X, θ, st)
    L = Float32(logitcrossentropy(Ŷ, Y))
    ws.fₒ[j] = 1.0f0 / (1.0f0 + L)
    return
end

function evaluate_fitness!(opt, ops::LEEAState{<:Matrix}, ws, model, st, X, Y)
    Threads.@threads for j in 1:opt.N
        evaluate_individual!(ops, ws, model, st, X, Y, j)
    end
    return (1.0f0 / maximum(ws.fₒ)) - 1.0f0
end

function evaluate_fitness!(opt, ops::LEEAState, ws, model, st, X, Y)
    for j in 1:opt.N
        evaluate_individual!(ops, ws, model, st, X, Y, j)
    end
    return (1.0f0 / maximum(ws.fₒ)) - 1.0f0
end

function inherit_fitness!(opt, ops, ws)
    d′ = 1.0f0 - opt.d
    half_d′ = 0.5f0 * d′
    Nₐ = length(ws.pₐ)

    for j in 1:opt.N
        if ops.is_first_step
            continue
        elseif j <= Nₐ
            ws.fₒ[j] += ops.fₚ[ws.pₐ[j]] * d′
        else
            ws.fₒ[j] += (ops.fₚ[ws.p₁[j - Nₐ]] + ops.fₚ[ws.p₂[j - Nₐ]]) * half_d′
        end
    end
    return
end

function select_parents!(opt, ws, rng)
    wheel = partialsortperm(ws.fₒ, 1:round(Int, opt.p * opt.N), rev = true)
    weights = Weights(@view ws.fₒ[wheel])

    sample!(rng, wheel, weights, ws.pₐ)
    sample!(rng, wheel, weights, ws.p₁)
    sample!(rng, wheel, weights, ws.p₂)

    collisions = ws.p₁ .== ws.p₂
    while any(collisions)
        n_collisions = sum(collisions)
        ws.p₂[collisions] .= sample(rng, wheel, weights, n_collisions)
        collisions = ws.p₁ .== ws.p₂
    end
    return
end

function reproduce_assexual!(opt, ops, ws, rng)
    θ_len = size(ops.P, 1)
    Nₐ = length(ws.pₐ)

    pₐ = similar(ops.P, Int, Nₐ)
    copyto!(pₐ, ws.pₐ)

    u₁ = similar(ops.P, Float32, θ_len, Nₐ)
    u₂ = similar(ops.P, Float32, θ_len, Nₐ)
    rand!(u₁)
    rand!(u₂)

    @views ws.O[:, 1:Nₐ] .= ops.P[:, pₐ] .+ (u₁ .< opt.r) .* ops.m .* (2.0f0 .* u₂ .- 1.0f0)
    return
end

function reproduce_sexual!(ops, ws, rng)
    θ_len = size(ops.P, 1)
    Nₐ = length(ws.pₐ)
    Nₛ = length(ws.p₁)

    p₁ = similar(ops.P, Int, Nₛ)
    p₂ = similar(ops.P, Int, Nₛ)
    copyto!(p₁, ws.p₁)
    copyto!(p₂, ws.p₂)

    u = similar(ops.P, Float32, θ_len, Nₛ)
    rand!(u)

    @views ws.O[:, (Nₐ + 1):end] .= ifelse.(u .< 0.5f0, ops.P[:, p₁], ops.P[:, p₂])
    return
end

function step!(opt::LEEA, ops, ws, model, st, X, Y, rng)
    best_loss = evaluate_fitness!(opt, ops, ws, model, st, X, Y)
    inherit_fitness!(opt, ops, ws)
    select_parents!(opt, ws, rng)
    reproduce_assexual!(opt, ops, ws, rng)
    reproduce_sexual!(ops, ws, rng)

    ops.P, ws.O = ws.O, ops.P
    ops.fₚ, ws.fₒ = ws.fₒ, ops.fₚ
    ops.is_first_step = false

    return best_loss
end

function get_best_params(ops::LEEAState)
    best_idx = argmax(ops.fₚ)
    return ops.re(@view ops.P[:, best_idx])
end

function update_scheduler!(opt::LEEA, ops, acc, best_acc)
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

format_metrics(ops::LEEAState) = @sprintf("      m = %.4f", ops.m)
