const logitcrossentropy = Lux.CrossEntropyLoss(; logits = Val(true))

@kwdef struct Experiment{
        O <: AbstractOptimizer, M <: Lux.AbstractLuxLayer, D <: AbstractDevice,
    }
    seed::Int = 42
    batchsize::Int = 100
    max_i::Int = 100000
    target_acc::Float64 = 100.0
    val_freq::Int = 10
    opt::O = LEEA()
    model::M = CNN_2C2D_MNIST
    device::D = Lux.gpu_device()
end

mutable struct ExperimentState{R <: AbstractRNG, O <: AbstractOptimizerState}
    rng::R
    ops::O
    last_checkpoint::Union{String, Nothing}
    best_acc::Float64
    i::Int
end

function init(exp::Experiment, model)
    LuxCUDA.CUDA.seed!(exp.seed)
    rng = Xoshiro(exp.seed)
    ops = init(exp.opt, model, exp.device, rng)
    return ExperimentState(rng, ops, nothing, 0.0, 1)
end

function evaluate(θ, model, st, val_set)
    X, Y = val_set
    Ŷ, _ = model(X, θ, st)

    correct = sum(onecold(Array(Ŷ), 0:9) .== Y)
    total = length(Y)

    return (correct / total) * 100.0
end

function save_checkpoint!(est, exp)
    opt_name = nameof(typeof(exp.opt))
    acc_int = est.best_acc * 100
    time_str = Dates.format(Dates.now(), "yyyy-mm-ddTHHMMSS")

    base_name = @sprintf("%s_%ia_%ii_%s.jls", opt_name, acc_int, est.i, time_str)

    filepath = joinpath("checkpoints", base_name)
    mkpath(dirname(filepath))

    if !isnothing(est.last_checkpoint)
        rm(est.last_checkpoint)
    end
    est.last_checkpoint = filepath

    return serialize(filepath, est)
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

        L = step!(exp.opt, est.ops, ws, model, st, X, Y, est.rng)
        Δt = time() - t₀
        opt_metrics = format_metrics(est.ops)

        base_log = @sprintf(
            "i = %-*d      Δt = %.2fs      L = %.4f%s",
            ndigits(exp.max_i), est.i, Δt, L, opt_metrics
        )

        if est.i % exp.val_freq == 0 || est.i == exp.max_i
            θ = get_best_params(est.ops)
            acc = evaluate(θ, model, st, val_set)

            update_scheduler!(exp.opt, est.ops, acc, est.best_acc)

            full_log = @sprintf("%s      Acc. = %-*.2f%%", base_log, 5, acc)

            if acc > est.best_acc
                println(full_log)
                est.best_acc = acc
                save_checkpoint!(est, exp)
            else
                @printf("%s [Best: %-*.2f%%]\n", full_log, 5, est.best_acc)
            end
        else
            println(base_log)
        end

        est.i += 1
    end
    return
end
