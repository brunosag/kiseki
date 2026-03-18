const logitcrossentropy = Lux.CrossEntropyLoss(; logits = Val(true))

@kwdef struct Experiment
    seed::Int = 42
    batchsize::Int = 500
    max_i::Int = 500000
    target_acc::Float64 = 100.0
    opt::AbstractOptimizer = LEEA()
    model::Lux.AbstractLuxLayer = CNN_2C2D_MNIST
    device::AbstractDevice = Lux.cpu_device()
end

mutable struct ExperimentState
    rng
    ops
    train_loader
    val_set
    test_loader
    best_acc
    i
end

function init(exp::Experiment)
    LuxCUDA.CUDA.seed!(exp.seed)

    rng = Xoshiro(exp.seed)
    ops = init(exp.opt, exp.model, exp.device, rng)
    train_loader, val_set, test_loader = load_MNIST(
        rng, exp.batchsize, exp.device, val_size = 10_000
    )
    best_acc = 0.0
    i = 1

    return ExperimentState(rng, ops, train_loader, val_set, test_loader, best_acc, i)
end

function evaluate(θ, model, st, val_set)
    X, Y = val_set
    Ŷ, _ = model(X, θ, st)

    correct = sum(onecold(Array(Ŷ), 0:9) .== Y)
    total = length(Y)

    return (correct / total) * 100.0
end

function run(exp::Experiment; est::Union{ExperimentState, Nothing} = nothing)
    st = Lux.testmode(Lux.initialstates(TaskLocalRNG(), exp.model))

    if isnothing(est)
        est = init(exp)
    end

    while est.i <= exp.max_i && est.best_acc < exp.target_acc
        t₀ = time()
        X, Y = popfirst!(est.train_loader) |> exp.device

        L = step!(exp.opt, est.ops, exp.model, st, X, Y, est.rng)

        θ = get_best_params(est.ops)
        acc = evaluate(θ, exp.model, st, est.val_set)

        update_scheduler!(exp.opt, est.ops, acc, est.best_acc)

        Δt = time() - t₀
        base_log = @sprintf "i = %-*d      Δt = %.2fs      L = %.4f      Acc. = %-*.2f%%" ndigits(exp.max_i) est.i Δt L 5 acc

        if acc > est.best_acc
            println(base_log)
            est.best_acc = acc
        else
            @printf "%s [Best: %-*.2f%%]\n" base_log 5 est.best_acc
        end

        est.i += 1
    end
    return
end
