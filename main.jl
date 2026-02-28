using Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Printf, Evolutionary, ComponentArrays, Serialization
using MLDatasets: MNIST
using SimpleChains: SimpleChains


function load_MNIST(rng; batchsize::Int)
    preprocess(data) = (
        reshape(data.features, 28, 28, 1, :),
        onehotbatch(data.targets, 0:9),
    )

    X_train, y_train = preprocess(MNIST(:train))
    X_test, y_test = preprocess(MNIST(:test))

    return (
        DataLoader((X_train, y_train); batchsize, shuffle = true, rng),
        DataLoader((X_test, y_test); batchsize, shuffle = false),
    )
end


function accuracy(model, θ, s, dataloader)
    s_test = Lux.testmode(s)
    correct, total = 0, 0

    for (X, y) in dataloader
        pred = Array(first(model(X, θ, s_test)))
        correct += sum(onecold(pred) .== onecold(y))
        total += size(y, 2)
    end

    return (correct / total) * 100
end


# GRADIENT IGNORED FOR NOW -------------------------------------------
function train_gradient(
        model;
        rng = Random.default_rng(),
        batchsize = 128,
        n_epochs = 10,
        optimizer = Adam(3.0f-4),
        lossfn = CrossEntropyLoss(; logits = Val(true)),
        ad_backend = AutoZygote(),
    )
    train_dataloader, test_dataloader = load_MNIST(rng; batchsize)
    θ, s = Lux.setup(rng, model)
    ts = Training.TrainState(model, θ, s, optimizer)

    for epoch in 1:n_epochs
        Δt = @elapsed begin
            for (X, y) in train_dataloader
                grads, loss, stats, ts = Training.single_train_step!(
                    ad_backend, lossfn, (X, y), ts
                )
            end
        end

        if epoch == 1 || epoch % 5 == 0 || epoch == n_epochs
            train_acc = accuracy(
                model, ts.parameters, ts.states, train_dataloader
            )
            test_acc = accuracy(
                model, ts.parameters, ts.states, test_dataloader
            )
            @printf "[%2d/%2d] \t Time %.2fs \t Training Accuracy: %.2f%% \t Test Accuracy: %.2f%%\n" epoch n_epochs Δt train_acc test_acc
        end
    end

    return ts
end
# --------------------------------------------------------------------


function Evolutionary.trace!(record::Dict{String, Any}, objfun, state, population, method::ES, options)
    best_idx = argmin(state.fitness)
    record["σ"] = copy(state.strategies[best_idx].σ)
    record["θ"] = copy(state.fittest)
    record["L"] = state.fitness[best_idx]
    return
end


function train_evolution(
        model;
        rng = Random.default_rng(),
        batchsize = 2048,
        iterations = 10000,
        lossfn = CrossEntropyLoss(; logits = Val(true)),
        checkpoint_interval = 100
    )
    train_dataloader, test_dataloader = load_MNIST(rng; batchsize)
    θ, s = Lux.setup(rng, model)
    s = Lux.testmode(s)

    θ_flat = ComponentArray(θ)
    x₀ = Vector(θ_flat)
    N = length(x₀)

    X, y = first(train_dataloader)

    function objective(x)
        θ_current = ComponentArray(x, getaxes(θ_flat))
        ŷ, _ = model(X, θ_current, s)
        return lossfn(ŷ, y)
    end

    complete_trace = Dict{String, Any}[]
    θ_latest = copy(x₀)
    it_counter = 0

    function checkpoint_callback(trace_record)
        it_counter += 1
        trace_dict = trace_record.metadata

        @printf(
            "Iter %5d \t Loss: %.6f\n", it_counter, trace_dict["L"]
        )

        push!(complete_trace, copy(trace_dict))
        θ_latest = trace_dict["θ"]

        if it_counter % checkpoint_interval == 0
            θ_current = ComponentArray(
                trace_dict["θ"], getaxes(θ_flat)
            )

            train_acc = accuracy(
                model, θ_current, s, train_dataloader
            )
            test_acc = accuracy(
                model, θ_current, s, test_dataloader
            )

            checkpoint_data = Dict(
                "i" => it_counter,
                "L" => trace_dict["L"],
                "θ" => trace_dict["θ"],
                "σ" => trace_dict["σ"],
                "train_acc" => train_acc,
                "test_acc" => test_acc
            )
            filename = @sprintf(
                "neuroevolution_checkpoint_%05d.jls", it_counter
            )
            serialize(filename, checkpoint_data)
            @printf(
                "\n[Checkpoint] Iter %d \t Loss: %.4f \t Train Acc: %.2f%% \t Test Acc: %.2f%%\n",
                it_counter, trace_dict["L"], train_acc, test_acc
            )
        end

        return false
    end

    options = Evolutionary.Options(
        iterations = iterations,
        show_every = 1,
        parallelization = :thread,
        rng = rng,
        abstol = -1.0,
        reltol = -1.0,
        successive_f_tol = 0,
        callback = checkpoint_callback,
    )

    μ = 20
    λ = 100
    poplt = [x₀ .+ 0.01f0 .* randn(rng, Float32, N) for _ in 1:μ]
    σ₀ = fill(0.01f0, N)
    τ = Float32(1.0 / sqrt(2.0 * sqrt(N)))
    τ′ = Float32(1.0 / sqrt(2.0 * N))

    algo = ES(
        initStrategy = AnisotropicStrategy(σ₀, τ, τ′),
        recombination = Evolutionary.average,
        srecombination = Evolutionary.average,
        mutation = Evolutionary.gaussian,
        smutation = Evolutionary.gaussian,
        μ = μ,
        λ = λ,
    )

    local result = nothing
    try
        result = Evolutionary.optimize(objective, Evolutionary.NoConstraints(), algo, poplt, options)
    catch e
        if e isa InterruptException
            @printf("\n[Interrupt] Neuroevolutionary optimization halted at iteration %d. Executing finishing steps...\n", it_counter)
        else
            rethrow(e)
        end
    end

    serialize("neuroevolution_complete_trace.jls", complete_trace)

    θ_best = ComponentArray(θ_latest, getaxes(θ_flat))
    train_acc = accuracy(model, θ_best, s, train_dataloader)
    test_acc = accuracy(model, θ_best, s, test_dataloader)

    @printf "\nFinal Training Accuracy: %.2f%% \t Test Accuracy: %.2f%%\n" train_acc test_acc

    return θ_best, result
end


model = ToSimpleChainsAdaptor((28, 28, 1))(
    Chain(
        Conv((3, 3), 1 => 8, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu),
        MaxPool((2, 2)),
        FlattenLayer(3),
        Dense(5 * 5 * 16 => 120, relu),
        Dense(120 => 10)
    )
)

rng = Xoshiro(42)
θ, result = train_evolution(model; rng)
