module Training

using Lux
using Random
using Printf
using Polyester
using ComponentArrays
using Statistics
using ..Data: load_MNIST
using ..Evaluation: accuracy
using ..Checkpoints: load_checkpoint
using ..Callbacks: CheckpointCallback

export train_evolution, ESConfig

Base.@kwdef struct ESConfig
    μ::Int = 100
    λ::Int = 500
    ema_decay::Float32 = 0.99f0
end

function select_population!(
        pop_parents, str_parents, fitness_parents,
        pop_offspring, str_offspring, fitness_offspring,
        buffers, config::ESConfig
    )
    μ = config.μ
    partialsortperm!(buffers.sort_idx, fitness_offspring, 1:μ)
    pop_parents .= pop_offspring[:, buffers.sort_idx[1:μ]]
    str_parents .= str_offspring[buffers.sort_idx[1:μ]]
    fitness_parents .= fitness_offspring[buffers.sort_idx[1:μ]]
    return
end

function train_evolution(
        model; I = 10000, batchsize = 1024, checkpoint_Δi = 10,
        resume_file = nothing, lossfn = CrossEntropyLoss(; logits = Val(true)),
        es_config = ESConfig(), save_dir = pwd(), rng = Random.default_rng()
    )
    train_dataloader, test_dataloader = load_MNIST(rng; batchsize)
    θ, s = Lux.setup(rng, model)
    s = Lux.testmode(s)
    θ_flat = ComponentArray(θ)
    N = length(θ_flat)
    axes_flat = getaxes(θ_flat)

    TraceType = NamedTuple{(:i, :L, :σ), Tuple{Int, Float32, Float32}}
    complete_trace = TraceType[]
    accuracy_trace = Tuple{Int, Float64}[]
    best_test_acc = 0.0
    i₀ = 0

    checkpoint_data = isnothing(resume_file) ? nothing : load_checkpoint(resume_file)
    if !isnothing(checkpoint_data)
        θ_latest = checkpoint_data["θ"]
        σ_latest = Float32(checkpoint_data["σ"])
        i₀ = checkpoint_data["i"]
        best_test_acc = get(checkpoint_data, "test_acc", 0.0)
        complete_trace = get(checkpoint_data, "complete_trace", complete_trace)
        accuracy_trace = get(checkpoint_data, "accuracy_trace", accuracy_trace)
        θ_ema = get(checkpoint_data, "θ_ema", copy(θ_latest))
    else
        θ_latest = Vector{Float32}(θ_flat)
        σ_latest = 0.01f0
        θ_ema = copy(θ_latest)
    end

    λ, μ = es_config.λ, es_config.μ

    pop_parents = repeat(θ_latest, 1, μ)
    str_parents = fill(σ_latest, μ)
    fitness_parents = fill(Inf32, μ)

    pop_offspring = Matrix{Float32}(undef, N, λ)
    str_offspring = Vector{Float32}(undef, λ)
    fitness_offspring = zeros(Float32, λ)

    buffers = (
        sort_idx = collect(1:λ),
    )

    τ = Float32(1.0 / sqrt(N))

    prev_checkpoint = Ref{String}(isnothing(resume_file) ? "" : resume_file)
    cb_state = CheckpointCallback(
        i₀, 0, checkpoint_Δi, time(), complete_trace, accuracy_trace,
        model, s, axes_flat, test_dataloader,
        best_test_acc, prev_checkpoint, save_dir, i₀
    )

    train_iter = iterate(train_dataloader)
    total_blocks = (I - i₀) ÷ checkpoint_Δi

    for block in 1:total_blocks
        if train_iter === nothing
            train_iter = iterate(train_dataloader)
        end
        (X, y), dl_state = train_iter
        train_iter = iterate(train_dataloader, dl_state)

        for inner_i in 1:checkpoint_Δi
            θ_avg = dropdims(sum(pop_parents, dims = 2), dims = 2) ./ Float32(μ)
            σ_avg = mean(str_parents)

            ϵ_0 = randn(rng, Float32, λ)
            ϵ_θ = randn(rng, Float32, N, λ)

            str_offspring .= σ_avg .* exp.(τ .* ϵ_0)

            pop_offspring .= θ_avg .+ reshape(str_offspring, 1, λ) .* ϵ_θ

            @batch for j in 1:λ
                θ_ind = ComponentArray(@view(pop_offspring[:, j]), axes_flat)
                ŷ, _ = model(X, θ_ind, s)
                fitness_offspring[j] = lossfn(ŷ, y)
            end

            select_population!(
                pop_parents, str_parents, fitness_parents,
                pop_offspring, str_offspring, fitness_offspring,
                buffers, es_config
            )

            θ_ema .= es_config.ema_decay .* θ_ema .+ (1.0f0 - es_config.ema_decay) .* pop_parents[:, 1]

            trace_record = (
                iteration = inner_i,
                metadata = (
                    θ = copy(pop_parents[:, 1]),
                    σ = str_parents[1],
                    L = fitness_parents[1],
                    θ_ema = copy(θ_ema),
                ),
            )

            cb_state.block = block
            cb_state(trace_record)
        end
    end

    return cb_state.complete_trace
end

end
