const logitcrossentropy = Lux.CrossEntropyLoss(; logits=Val(true))

@kwdef struct Experiment{O<:AbstractOptimizer,M<:Lux.AbstractLuxLayer,D<:AbstractDevice}
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

@kwdef mutable struct TrainingHistory
    loss::Vector{Float32} = Float32[]
    acc::Vector{NamedTuple{(:i, :value),Tuple{Int,Float64}}} = NamedTuple{(:i, :value),Tuple{Int,Float64}}[]
end

mutable struct ExperimentState{R<:AbstractRNG,O<:AbstractOptimizerState,C<:Tuple}
    rng::R
    ops::O
    last_checkpoint::Union{String,Nothing}
    best_acc::Float64
    i::Int
    callbacks::C
    history::TrainingHistory
end

function init(exp::Experiment, injected_callbacks=())
    LuxCUDA.CUDA.seed!(exp.seed)
    rng = Xoshiro(exp.seed)
    ops = init(exp.opt, exp.model, exp.device, rng)
    callbacks = (Tracker(), ConsoleLogger(), CheckpointSaver(), injected_callbacks...)
    return ExperimentState(rng, ops, nothing, 0.0, 1, callbacks, TrainingHistory())
end

function evaluate(θ, model, st, val_set)
    X, y = val_set
    Ŷ, _ = model(X, θ, st)

    ŷ = onecold(Array(Ŷ), 0:9)
    correct = sum(ŷ .== y)
    total = length(y)

    return round((correct / total) * 100.0, digits=2)
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
        rm(est.last_checkpoint * ".jls", force=true)
        rm(est.last_checkpoint * ".json", force=true)
    end
    est.last_checkpoint = joinpath("checkpoints", base_name)
    serialize(filepath_jls, est)

    metadata = Dict(
        "id" => base_name,
        "optimizer" => opt_name,
        "iteration" => est.i,
        "best_accuracy" => est.best_acc,
        "timestamp" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
        "seed" => exp.seed,
        "batchsize" => exp.batchsize,
        "hyperparameters" => get_hyperparams(exp.opt),
    )
    write(filepath_json, JSON3.write(metadata))
    return
end

function load_checkpoint(filepath, injected_callbacks=())
    est = deserialize(filepath)

    if isempty(injected_callbacks)
        return est
    end

    return ExperimentState(
        est.rng,
        est.ops,
        est.last_checkpoint,
        est.best_acc,
        est.i,
        (est.callbacks..., injected_callbacks...),
        est.history
    )
end

fastforward(loader, i) = foreach(_ -> popfirst!(loader), 1:i)

function run!(exp::Experiment, est::ExperimentState; resume=false)
    model = adapt_model(exp.model, exp.device)
    st = Lux.testmode(Lux.initialstates(TaskLocalRNG(), model))

    rng_data = Xoshiro(exp.seed)
    train_loader, val_set, _ = load_MNIST(
        rng_data, exp.batchsize, exp.device, val_size=10_000
    )

    if resume
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

            GC.gc(true)
            LuxCUDA.CUDA.reclaim()
        end

        est.i += 1
    end
    return
end
