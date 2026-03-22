@kwdef struct LEEA <: AbstractOptimizer
    N::Int = 1000        # population size
    r::Float32 = 0.04    # mutation rate
    m₀::Float32 = 0.03   # initial mutation power
    γ::Float32 = 0.99    # mutation power decay
    p::Float32 = 0.4     # selection proportion
    s::Float32 = 0.5     # sexual reproduction proportion
    d::Float32 = 0.2     # fitness inheritance decay
    pat_lim::Int = 5     # generations to wait before decaying m
end

mutable struct LEEAState{M <: AbstractMatrix{Float32}, R} <: AbstractOptimizerState
    re::R
    P::M
    fₚ::Vector{Float32}
    pₐ::Vector{Int}
    p₁::Vector{Int}
    p₂::Vector{Int}
    m::Float32
    pat::Int
    is_first_step::Bool
end

mutable struct LEEAWorkspace{M <: AbstractMatrix{Float32}}
    O::M
    fₒ::Vector{Float32}
end

function init(opt::LEEA, model, dev, rng)
    _, re = destructure(Lux.initialparameters(TaskLocalRNG(), model))
    P = stack([destructure(Lux.initialparameters(rng, model))[1] for _ in 1:opt.N]) |> dev

    Nₛ = round(Int, opt.s * opt.N)
    Nₐ = opt.N - Nₛ

    fₚ = Vector{Float32}(undef, opt.N)
    pₐ, p₁, p₂ = Vector{Int}(undef, Nₐ), Vector{Int}(undef, Nₛ), Vector{Int}(undef, Nₛ)

    return LEEAState(re, P, fₚ, pₐ, p₁, p₂, opt.m₀, 0, true)
end

function init_workspace(opt::LEEA, ops)
    O = similar(ops.P)
    fₒ = Vector{Float32}(undef, opt.N)
    return LEEAWorkspace(O, fₒ)
end

@inline function predict_individual(ops, model, st, X, j)
    θ = ops.re(@view ops.P[:, j])
    Ŷ, _ = model(X, θ, st)
    return Ŷ
end

function compute_fitness!(ws, Ŷ_pop, Y)
    Y_reshaped = reshape(Y, size(Y, 1), size(Y, 2), 1)
    L = dropdims(mean(-sum(Y_reshaped .* logsoftmax(Ŷ_pop; dims = 1); dims = 1); dims = 2); dims = (1, 2))

    copyto!(ws.fₒ, 1.0f0 ./ (1.0f0 .+ Array(L)))
    return (1.0f0 / maximum(ws.fₒ)) - 1.0f0
end

function evaluate_population!(opt, ops::LEEAState{<:Matrix}, ws, model, st, X, Y)
    Ŷ_list = Vector{Matrix{Float32}}(undef, opt.N)
    Threads.@threads for j in 1:opt.N
        Ŷ_list[j] = predict_individual(ops, model, st, X, j)
    end
    return compute_fitness!(ws, stack(Ŷ_list), Y)
end

function evaluate_population!(opt, ops::LEEAState, ws, model, st, X, Y)
    Ŷ_pop = stack([predict_individual(ops, model, st, X, j) for j in 1:opt.N])
    return compute_fitness!(ws, Ŷ_pop, Y)
end

function inherit_fitness!(opt, ops, ws)
    d′ = 1.0f0 - opt.d
    half_d′ = 0.5f0 * d′
    Nₐ = length(ops.pₐ)

    for j in 1:opt.N
        if ops.is_first_step
            continue
        elseif j <= Nₐ
            ws.fₒ[j] += ops.fₚ[ops.pₐ[j]] * d′
        else
            ws.fₒ[j] += (ops.fₚ[ops.p₁[j - Nₐ]] + ops.fₚ[ops.p₂[j - Nₐ]]) * half_d′
        end
    end
    return
end

function select_parents!(opt, ops, ws, rng)
    wheel = partialsortperm(ws.fₒ, 1:round(Int, opt.p * opt.N), rev = true)
    weights = Weights(@view ws.fₒ[wheel])

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

function reproduce_assexual!(opt, ops, ws, rng)
    θ_len = size(ops.P, 1)
    Nₐ = length(ops.pₐ)

    pₐ = similar(ops.P, Int, Nₐ)
    copyto!(pₐ, ops.pₐ)

    u₁ = similar(ops.P, Float32, θ_len, Nₐ)
    u₂ = similar(ops.P, Float32, θ_len, Nₐ)
    rand!(u₁)
    rand!(u₂)

    @views ws.O[:, 1:Nₐ] .= ops.P[:, pₐ] .+ (u₁ .< opt.r) .* ops.m .* (2.0f0 .* u₂ .- 1.0f0)
    return
end

function reproduce_sexual!(ops, ws, rng)
    θ_len = size(ops.P, 1)
    Nₐ = length(ops.pₐ)
    Nₛ = length(ops.p₁)

    p₁ = similar(ops.P, Int, Nₛ)
    p₂ = similar(ops.P, Int, Nₛ)
    copyto!(p₁, ops.p₁)
    copyto!(p₂, ops.p₂)

    u = similar(ops.P, Float32, θ_len, Nₛ)
    rand!(u)

    @views ws.O[:, (Nₐ + 1):end] .= ifelse.(u .< 0.5f0, ops.P[:, p₁], ops.P[:, p₂])
    return
end

function step!(opt::LEEA, ops, ws, model, st, X, Y, rng)
    best_loss = evaluate_population!(opt, ops, ws, model, st, X, Y)
    inherit_fitness!(opt, ops, ws)
    select_parents!(opt, ops, ws, rng)
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

function update_scheduler!(opt::LEEA, ops, is_best)
    if is_best
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
