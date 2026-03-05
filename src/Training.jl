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
end

Base.@kwdef struct GradientConfig
    α::Float32 = 3.0f-4
    epochs::Int = 10
    batchsize::Int = 128
end


function initialize_training_state(
        model, resume_file, save_dir, I, checkpoint_Δi, batchsize, rng, prefix, verbose
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
        prev_checkpoint, save_dir, prefix, verbose
    )

    return train_dataloader, θ_flat, s, checkpoint_data, i₀, cb_state
end


function train_gradient(
        model; config = GradientConfig(), checkpoint_Δi = 100,
        resume_file = nothing, lossfn = CrossEntropyLoss(; logits = Val(true)),
        save_dir = pwd(), rng = Random.default_rng()
    )

    train_dataloader_temp, _ = load_MNIST(rng; batchsize = config.batchsize)
    total_iterations = config.epochs * length(train_dataloader_temp)

    train_dataloader, θ_flat, s, checkpoint_data, i₀, cb_state = initialize_training_state(
        model, resume_file, save_dir, total_iterations, checkpoint_Δi, config.batchsize, rng, "Adam", false
    )

    dev = Lux.gpu_device()
    cpu_dev = Lux.cpu_device()

    θ_structured = ComponentArray(
        isnothing(checkpoint_data) ? θ_flat : checkpoint_data["θ"],
        cb_state.axes
    )

    optimizer = Optimisers.Adam(config.α)
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

    if i₀ > 0
        println("Fast-forwarding dataloader to iteration $i₀...")
    end

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

        cb_state(global_i, ts.parameters, Float32(cpu_dev(loss)), 0.0f0, ts.optimizer_state)
    end

    return cb_state.complete_trace
end


function train_evolution(
        model;
        I = 500000, batchsize = 1024, checkpoint_Δi = 10,
        resume_file = nothing, lossfn = CrossEntropyLoss(; logits = Val(true)),
        es_config = ESConfig(), save_dir = pwd(), rng = Random.default_rng()
    )

    train_dataloader, θ_flat, s, checkpoint_data, i₀, cb_state = initialize_training_state(
        model, resume_file, save_dir, I, checkpoint_Δi, batchsize, rng, "ES", false
    )

    dev = Lux.gpu_device()
    cpu_dev = Lux.cpu_device()

    N = length(θ_flat)
    λ, μ = es_config.λ, es_config.μ
    τ = Float32(1.0 / sqrt(N))

    θ_latest = isnothing(checkpoint_data) ? Vector{Float32}(θ_flat) : checkpoint_data["θ"]
    σ_latest = isnothing(checkpoint_data) ? 0.01f0 : Float32(checkpoint_data["σ"])

    if !isnothing(checkpoint_data) && haskey(checkpoint_data, "opt_state")
        es_state = checkpoint_data["opt_state"]
        pop_parents = dev(es_state.pop_parents)
        str_parents = dev(es_state.str_parents)
        fitness_parents = es_state.fitness_parents
    else
        pop_parents = dev(repeat(θ_latest, 1, μ))
        str_parents = dev(fill(σ_latest, μ))
        fitness_parents = fill(Inf32, μ)
    end

    fitness_offspring = zeros(Float32, λ)

    pop_offspring = dev(Matrix{Float32}(undef, N, λ))
    str_offspring = dev(Vector{Float32}(undef, λ))

    ϵ_0 = dev(Vector{Float32}(undef, λ))
    ϵ_θ = dev(Matrix{Float32}(undef, N, λ))

    sort_idx_cpu = collect(1:λ)
    best_idx_cpu_buffer = zeros(Int, μ)
    best_idx = dev(collect(1:μ))
    s_dev = dev(s)

    if i₀ == 0
        es_state = (pop_parents = pop_parents, str_parents = str_parents, fitness_parents = fitness_parents)
        cb_state(i₀, @view(pop_parents[:, 1]), fitness_parents[1], cpu_dev(str_parents[1:1])[1], es_state)
    end

    data_iter = Iterators.Stateful(Iterators.cycle(train_dataloader))

    if i₀ > 0
        println("Fast-forwarding dataloader and PRNG state to iteration $i₀...")
    end

    for _ in 1:i₀
        popfirst!(data_iter)
        randn!(ϵ_0)
        randn!(ϵ_θ)
    end

    for i in (i₀ + 1):I
        (X, y) = popfirst!(data_iter)
        X_dev, y_dev = dev(X), dev(y)

        θ_avg = dropdims(sum(pop_parents, dims = 2), dims = 2) ./ Float32(μ)
        σ_avg = mean(str_parents)

        randn!(ϵ_0)
        randn!(ϵ_θ)

        @. str_offspring = σ_avg * exp(τ * ϵ_0)
        pop_offspring .= θ_avg .+ reshape(str_offspring, 1, λ) .* ϵ_θ

        for j in 1:λ
            θ_ind = ComponentArray(@view(pop_offspring[:, j]), cb_state.axes)
            ŷ, _ = model(X_dev, θ_ind, s_dev)
            fitness_offspring[j] = cpu_dev(lossfn(ŷ, y_dev))[1]
        end

        partialsortperm!(sort_idx_cpu, fitness_offspring, 1:μ)
        copyto!(best_idx_cpu_buffer, @view sort_idx_cpu[1:μ])
        copyto!(best_idx, best_idx_cpu_buffer)

        pop_parents .= @view pop_offspring[:, best_idx]
        str_parents .= @view str_offspring[best_idx]
        fitness_parents .= fitness_offspring[best_idx_cpu_buffer]

        es_state = (pop_parents = pop_parents, str_parents = str_parents, fitness_parents = fitness_parents)
        cb_state(i, @view(pop_parents[:, 1]), fitness_parents[1], cpu_dev(str_parents[1:1])[1], es_state)

        if i % 100 == 0
            GC.gc(false)
            LuxCUDA.CUDA.reclaim()
        end
    end

    return cb_state.complete_trace
end

end
