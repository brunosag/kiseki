module Training

using Lux, Random, Printf, ComponentArrays, Statistics, Optimisers, Zygote, LuxCUDA
using ..Data: load_MNIST
using ..Evaluation: accuracy
using ..Checkpoints: load_checkpoint
using ..Callbacks: CheckpointCallback

export train_gradient, GradientConfig, train_evolution, ESConfig


Base.@kwdef struct ESConfig
    μ::Int = 30
    λ::Int = 150
    β::Float32 = 0.9
    weight_decay::Float32 = 1.0f-4
end

Base.@kwdef struct GradientConfig
    η::Float32 = 0.001
    β::Tuple{Float32, Float32} = (0.9, 0.999)
    weight_decay::Float32 = 1.0f-4
end


function initialize_training_state(
        model, resume_file, save_dir, I, checkpoint_Δi, batchsize, rng, prefix, target_acc
    )

    train_dataloader, test_dataloader = load_MNIST(rng; batchsize)

    θ, s = Lux.setup(rng, model)
    s = Lux.testmode(s)

    θ_flat = ComponentArray(θ)
    axes_flat = getaxes(θ_flat)

    checkpoint_data = isnothing(resume_file) ? nothing : load_checkpoint(resume_file)

    i₀ = isnothing(checkpoint_data) ? 0 : checkpoint_data["i"]
    best_test_acc = isnothing(checkpoint_data) ? 0.0 : get(checkpoint_data, "test_acc", 0.0)

    TraceType = NamedTuple{(:i, :L, :σ), Tuple{Int, Float32, Float32}}
    complete_trace = isnothing(checkpoint_data) ? TraceType[] : get(checkpoint_data, "complete_trace", TraceType[])
    accuracy_trace = isnothing(checkpoint_data) ? Tuple{Int, Float64}[] : get(checkpoint_data, "accuracy_trace", Tuple{Int, Float64}[])

    prev_checkpoint = Ref{String}(isnothing(resume_file) ? "" : resume_file)

    cb_state = CheckpointCallback(
        I, checkpoint_Δi, time(), complete_trace, accuracy_trace,
        model, s, axes_flat, test_dataloader, best_test_acc,
        prev_checkpoint, save_dir, prefix, target_acc
    )

    return train_dataloader, θ_flat, s, checkpoint_data, i₀, cb_state
end


function train_gradient(
        model; epochs = 1000, batchsize = 128, config = GradientConfig(),
        checkpoint_Δi = 1, resume_file = nothing, rng = Random.default_rng(),
        lossfn = CrossEntropyLoss(; logits = Val(true)), save_dir = pwd(),
        target_acc::Float64 = 0.9872
    )

    train_dataloader_temp, _ = load_MNIST(rng; batchsize)
    total_iterations = epochs * length(train_dataloader_temp)

    train_dataloader, θ_flat, s, checkpoint_data, i₀, cb_state = initialize_training_state(model, resume_file, save_dir, total_iterations, checkpoint_Δi, batchsize, rng, "AdamW", target_acc)

    dev = Lux.gpu_device()
    cpu_dev = Lux.cpu_device()

    θ_structured = ComponentArray(
        isnothing(checkpoint_data) ? θ_flat : checkpoint_data["θ"],
        cb_state.axes
    )

    optimizer = Optimisers.AdamW(config.η, config.β, config.weight_decay)
    ts = dev(Lux.Training.TrainState(model, θ_structured, s, optimizer))

    if !isnothing(checkpoint_data) && haskey(checkpoint_data, "opt_state")
        ts = Lux.Training.TrainState(
            ts.cache, ts.objective_function, ts.allocator_cache,
            ts.model, ts.parameters, ts.states, ts.optimizer,
            dev(checkpoint_data["opt_state"]), ts.step
        )
    end

    ad_backend = AutoZygote()

    global_i = i₀
    data_iter = Iterators.Stateful(Iterators.cycle(train_dataloader))

    for _ in 1:i₀
        popfirst!(data_iter)
    end

    for _ in (i₀ + 1):total_iterations
        (X, y) = popfirst!(data_iter)

        X_dev, y_dev = dev(X), dev(y)
        global_i += 1

        grads, loss, stats, ts = Lux.Training.single_train_step!(
            ad_backend, lossfn, (X_dev, y_dev), ts
        )

        if cb_state(global_i, ts.parameters, Float32(cpu_dev(loss)), 0.0f0, ts.optimizer_state)
            break
        end
    end

    return cb_state.complete_trace
end


function train_evolution(
        model; I = 500000, batchsize = 1024, checkpoint_Δi = 10,
        resume_file = nothing, lossfn = CrossEntropyLoss(; logits = Val(true)),
        es_config = ESConfig(), save_dir = pwd(), rng = Random.default_rng(),
        target_acc::Float64 = 1.0
    )

    train_dataloader, θ_flat, s, checkpoint_data, i₀, cb_state = initialize_training_state(
        model, resume_file, save_dir, I, checkpoint_Δi, batchsize, rng, "ES", target_acc
    )

    dev = Lux.gpu_device()
    cpu_dev = Lux.cpu_device()

    N = length(θ_flat)
    λ, μ, β, weight_decay = es_config.λ, es_config.μ, es_config.β, es_config.weight_decay
    τ = Float32(1.0 / sqrt(N))

    w_cpu = Float32.(log(μ + 0.5) .- log.(1:μ))
    w_cpu ./= sum(w_cpu)
    w = dev(w_cpu)

    θ_latest = isnothing(checkpoint_data) ? Vector{Float32}(θ_flat) : checkpoint_data["θ"]
    σ_latest = isnothing(checkpoint_data) ? 0.01f0 : Float32(checkpoint_data["σ"])

    if !isnothing(checkpoint_data) && haskey(checkpoint_data, "opt_state")
        es_state = checkpoint_data["opt_state"]
        pop_parents = dev(es_state.pop_parents)
        str_parents = dev(es_state.str_parents)
        fitness_parents = es_state.fitness_parents
        θ_avg = dev(es_state.θ_avg)
        σ_avg = Float32(es_state.σ_avg)
    else
        pop_parents = dev(repeat(θ_latest, 1, μ))
        str_parents = dev(fill(σ_latest, μ))
        fitness_parents = fill(Inf32, μ)
        θ_avg = dev(copy(θ_latest))
        σ_avg = Float32(σ_latest)
    end

    fitness_offspring = zeros(Float32, λ)

    pop_offspring = dev(Matrix{Float32}(undef, N, λ))
    str_offspring = dev(Vector{Float32}(undef, λ))

    @assert λ % 2 == 0 "λ must be even for mirrored sampling"
    half_λ = λ ÷ 2

    ϵ_0_half = dev(Vector{Float32}(undef, half_λ))
    ϵ_θ_half = dev(Matrix{Float32}(undef, N, half_λ))

    ϵ_0 = dev(Vector{Float32}(undef, λ))
    ϵ_θ = dev(Matrix{Float32}(undef, N, λ))

    function sample_noise!()
        randn!(ϵ_0_half)
        randn!(ϵ_θ_half)
        @view(ϵ_0[1:half_λ]) .= ϵ_0_half
        @view(ϵ_0[(half_λ + 1):end]) .= .-ϵ_0_half
        @view(ϵ_θ[:, 1:half_λ]) .= ϵ_θ_half
        @view(ϵ_θ[:, (half_λ + 1):end]) .= .-ϵ_θ_half
        return nothing
    end

    sort_idx_cpu = collect(1:λ)
    s_dev = dev(s)

    if i₀ == 0
        es_state = (pop_parents = pop_parents, str_parents = str_parents, fitness_parents = fitness_parents, θ_avg = θ_avg, σ_avg = σ_avg)
        if cb_state(i₀, θ_avg, fitness_parents[1], σ_avg, es_state)
            return cb_state.complete_trace
        end
    end

    data_iter = Iterators.Stateful(Iterators.cycle(train_dataloader))

    for _ in 1:i₀
        popfirst!(data_iter)
        sample_noise!()
    end

    for i in (i₀ + 1):I
        (X, y) = popfirst!(data_iter)
        X_dev, y_dev = dev(X), dev(y)

        sample_noise!()

        @. str_offspring = σ_avg * exp(τ * ϵ_0)
        pop_offspring .= θ_avg .+ reshape(str_offspring, 1, λ) .* ϵ_θ

        for j in 1:λ
            θ_ind = ComponentArray(@view(pop_offspring[:, j]), cb_state.axes)
            ŷ, _ = model(X_dev, θ_ind, s_dev)

            base_loss = cpu_dev(lossfn(ŷ, y_dev))[1]
            l2_penalty = weight_decay * cpu_dev(sum(abs2, θ_ind))

            fitness_offspring[j] = base_loss + l2_penalty
        end

        partialsortperm!(sort_idx_cpu, fitness_offspring, 1:μ)
        best_idx_cpu = @view sort_idx_cpu[1:μ]
        best_idx = dev(best_idx_cpu)

        pop_parents .= pop_offspring[:, best_idx]
        str_parents .= str_offspring[best_idx]
        fitness_parents .= fitness_offspring[best_idx_cpu]

        θ_avg_target = pop_parents * w
        θ_avg .= β .* θ_avg .+ (1.0f0 - β) .* θ_avg_target

        σ_avg_target = sum(cpu_dev(str_parents) .* w_cpu)
        σ_avg = β * σ_avg + (1.0f0 - β) * σ_avg_target

        es_state = (pop_parents = pop_parents, str_parents = str_parents, fitness_parents = fitness_parents, θ_avg = θ_avg, σ_avg = σ_avg)
        if cb_state(i, θ_avg, fitness_parents[1], σ_avg, es_state)
            break
        end

        if i % 1000 == 0
            GC.gc(true)
            LuxCUDA.CUDA.reclaim()
        end
    end

    return cb_state.complete_trace
end


end
