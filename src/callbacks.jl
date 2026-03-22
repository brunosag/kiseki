abstract type AbstractCallback end

on_step_end!(cb::AbstractCallback, exp, est, loss, Δt) = nothing
on_val_end!(cb::AbstractCallback, exp, est, acc, is_best) = nothing

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

function on_val_end!(cb::ConsoleLogger, exp, est, acc, is_best)
    acc_log = @sprintf("      Acc. = %-*.2f%%", 5, acc)
    is_best ? println(acc_log) : @printf("%s [Best: %-*.2f%%]\n", acc_log, 5, est.best_acc)
    return
end

# ---------------- CheckpointSaver ----------------

struct CheckpointSaver <: AbstractCallback end

function on_val_end!(cb::CheckpointSaver, exp, est, acc, is_best)
    if is_best
        save_checkpoint!(est, exp)
    end
    return
end
