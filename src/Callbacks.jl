module Callbacks

using Printf, ComponentArrays, Statistics
using Lux: cpu_device
using ..Evaluation: accuracy
using ..Checkpoints: save_checkpoint

export CheckpointCallback

mutable struct CheckpointCallback{M, S, DL, T} <: Function
    I::Int
    checkpoint_Δi::Int
    i_timer::Float64
    complete_trace::Vector{T}
    accuracy_trace::Vector{Tuple{Int, Float64}}
    model::M
    st::S
    axes::Any
    test_dataloader::DL
    best_test_acc::Float64
    prev_checkpoint::Ref{String}
    save_dir::String
    prefix::String
    target_acc::Float64
end

function (cb::CheckpointCallback)(
        global_i::Int, θ_dev, L::Float32, σ::Float32, opt_state_dev = nothing
    )

    current_time = time()
    elapsed = current_time - cb.i_timer
    cb.i_timer = current_time

    push!(cb.complete_trace, (i = global_i, L = L, σ = σ))

    if global_i % cb.checkpoint_Δi == 0
        cpu_dev_fn = cpu_device()
        θ_cpu = Vector{Float32}(cpu_dev_fn(θ_dev))
        θ_current = ComponentArray(θ_cpu, cb.axes)

        raw_acc = accuracy(cb.model, θ_current, cb.st, cb.test_dataloader)
        test_acc = raw_acc > 1.0 ? raw_acc / 100.0 : raw_acc

        push!(cb.accuracy_trace, (global_i, test_acc))

        interval_start = max(1, length(cb.complete_trace) - cb.checkpoint_Δi + 1)
        interval_losses = [t.L for t in @view cb.complete_trace[interval_start:end]]
        loss_mean = mean(interval_losses)
        loss_std = length(interval_losses) > 1 ? std(interval_losses) : 0.0f0

        base_log = @sprintf(
            "i = %-*d      L = %-*.4f      mean(L) = %-*.4f      std(L) = %-*.4f      Δt = %.3fs      σ = %.4f",
            ndigits(cb.I), global_i, 8, L, 8, loss_mean, 8, loss_std, elapsed, σ
        )

        if test_acc > cb.best_test_acc
            cb.best_test_acc = test_acc

            checkpoint_data = Dict{String, Any}(
                "i" => global_i,
                "L" => L,
                "θ" => θ_cpu,
                "σ" => σ,
                "test_acc" => test_acc,
                "complete_trace" => cb.complete_trace,
                "accuracy_trace" => cb.accuracy_trace
            )

            if !isnothing(opt_state_dev)
                checkpoint_data["opt_state"] = cpu_dev_fn(opt_state_dev)
            end

            save_checkpoint(checkpoint_data, test_acc, global_i, cb.prev_checkpoint, cb.save_dir, cb.prefix)

            acc_str = @sprintf("Accuracy = %.2f%%", test_acc * 100.0)
        else
            acc_str = @sprintf("Accuracy = %.2f%% (Best: %.2f%%)", test_acc * 100.0, cb.best_test_acc * 100.0)
        end

        println(base_log, "      ", acc_str)

        if test_acc >= cb.target_acc
            return true
        end
    end

    return false
end

end
