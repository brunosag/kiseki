const logitcrossentropy = Lux.CrossEntropyLoss(; logits = Val(true))

@kwdef struct Experiment{O <: AbstractOptimizer, M <: Lux.AbstractLuxLayer, D <: AbstractDevice}
    opt::O = LEEA()
    device::D = Lux.gpu_device()
    model::M = CNN_2C2D_MNIST
    seed::Int = 42
    batchsize::Int = 500
    max_i::Int = 100000
    target_acc::Float64 = 100.0
    val_freq::Int = 10
    save_freq::Int = 50
end

mutable struct ExperimentState{R <: AbstractRNG, O <: AbstractOptimizerState, C <: AbstractVector{<:AbstractCallback}}
    rng::R
    ops::O
    last_checkpoint::Union{String, Nothing}
    best_acc::Float64
    i::Int
    callbacks::C
end

function init(exp::Experiment, model)
    LuxCUDA.CUDA.seed!(exp.seed)
    rng = Xoshiro(exp.seed)
    ops = init(exp.opt, model, exp.device, rng)
    callbacks = [Tracker(), ConsoleLogger(), CheckpointSaver()]
    return ExperimentState(rng, ops, nothing, 0.0, 1, callbacks)
end

function evaluate(θ, model, st, val_set)
    X, y = val_set
    Ŷ, _ = model(X, θ, st)

    ŷ = onecold(Array(Ŷ), 0:9)
    correct = sum(ŷ .== y)
    total = length(y)

    return (correct / total) * 100.0
end

get_hyperparams(opt) = Dict(string(f) => getproperty(opt, f) for f in propertynames(opt))

function save_checkpoint!(est, exp)
    opt_name = nameof(typeof(exp.opt))
    acc_int = est.best_acc * 100
    time_str = Dates.format(Dates.now(), "yyyy-mm-ddTHHMMSS")

    base_name = @sprintf("%s_%ia_%ii_%s", opt_name, acc_int, est.i, time_str)

    filepath_jls = joinpath("checkpoints", base_name * ".jls")
    filepath_json = joinpath("checkpoints", base_name * ".json")
    mkpath(dirname(filepath_jls))

    if !isnothing(est.last_checkpoint)
        rm(est.last_checkpoint * ".jls", force = true)
        rm(est.last_checkpoint * ".json", force = true)
    end
    est.last_checkpoint = joinpath("checkpoints", base_name)
    serialize(filepath_jls, est)

    metadata = Dict(
        "id" => base_name,
        "optimizer" => opt_name,
        "iteration" => est.i,
        "best_accuracy" => est.best_acc,
        "timestamp" => time_str,
        "seed" => exp.seed,
        "batchsize" => exp.batchsize,
        "hyperparameters" => get_hyperparams(exp.opt),
    )
    write(filepath_json, JSON3.write(metadata))
    return
end

load_checkpoint(filepath) = deserialize(filepath)

fastforward(loader, i) = foreach(_ -> popfirst!(loader), 1:i)

function run(exp, est = nothing)
    model = adapt_model(CNN_2C2D_MNIST, exp.device)
    st = Lux.testmode(Lux.initialstates(TaskLocalRNG(), model))
    rng_data = Xoshiro(exp.seed)
    train_loader, val_set, _ = load_MNIST(
        rng_data, exp.batchsize, exp.device, val_size = 10_000
    )

    if isnothing(est)
        est = init(exp, model)
    else
        fastforward(train_loader, est.i)
        est.i += 1
    end

    ws = init_workspace(exp.opt, est.ops)

    while est.i <= exp.max_i && est.best_acc < exp.target_acc
        t₀ = time()
        X, Y = popfirst!(train_loader)

        loss = step!(exp.opt, est.ops, ws, model, st, X, Y, est.rng)
        Δt = time() - t₀

        foreach(cb -> on_step_end!(cb, exp, est, loss, Δt), est.callbacks)

        if est.i % exp.val_freq == 0 || est.i == exp.max_i
            θ = get_best_params(est.ops)
            acc = evaluate(θ, model, st, val_set)

            is_best = acc > est.best_acc
            if is_best
                est.best_acc = acc
            end

            update_scheduler!(exp.opt, est.ops, is_best)

            foreach(cb -> on_val_end!(cb, exp, est, val_set, model, θ, st, acc, is_best), est.callbacks)
        end

        est.i += 1
    end
    return
end
