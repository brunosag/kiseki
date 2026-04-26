abstract type AbstractCallback end

on_step_end!(cb::AbstractCallback, exp, est, loss, Δt) = nothing
on_val_end!(cb::AbstractCallback, exp, est, val_set, model, θ, st, acc, is_best) = nothing

# ---------------- Tracker ----------------

@kwdef mutable struct Tracker <: AbstractCallback end

on_step_end!(cb::Tracker, exp, est, loss, Δt) = push!(est.history.loss, loss)
on_val_end!(cb::Tracker, exp, est, val_set, model, θ, st, acc, is_best) = push!(est.history.acc, (i=est.i, value=acc))

# ---------------- CheckpointSaver ----------------

struct CheckpointSaver <: AbstractCallback end

function on_step_end!(cb::CheckpointSaver, exp, est, loss, Δt)
    if est.i % exp.save_freq == 0 || est.i == exp.max_i
        save_checkpoint!(est, exp)
    end
    return
end

# ---------------- ConsoleLogger ----------------

struct ConsoleLogger <: AbstractCallback end

function on_step_end!(cb::ConsoleLogger, exp, est, loss, Δt)
    opt_metrics = format_metrics(est.ops)
    base_log = @sprintf(
        "i = %-*d      Δt = %.2fs      L = %.4f%s",
        ndigits(exp.max_i), est.i, Δt, loss, opt_metrics
    )
    if est.i % exp.val_freq == 0 || est.i == exp.max_i
        print(base_log)
    else
        println(base_log)
    end
    return
end

function on_val_end!(cb::ConsoleLogger, exp, est, val_set, model, θ, st, acc, is_best)
    acc_log = @sprintf("      Acc. = %-*.2f%%", 5, acc)
    is_best ? println(acc_log) : @printf("%s [Best: %-*.2f%%]\n", acc_log, 5, est.best_acc)
    return
end
