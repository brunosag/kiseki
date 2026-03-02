module Callbacks

using Printf
using ComponentArrays
using ..Evaluation: accuracy
using ..Checkpoints: save_checkpoint

export CheckpointCallback

mutable struct CheckpointCallback{M, S, DL, T} <: Function
    i₀::Int
    block::Int
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
    current_global_i::Int
end

function (cb::CheckpointCallback)(trace_record)
    trace_record.iteration == 0 && return false

    global_i = cb.i₀ + (cb.block - 1) * cb.checkpoint_Δi + trace_record.iteration
    cb.current_global_i = global_i

    current_time = time()
    elapsed = current_time - cb.i_timer
    cb.i_timer = current_time

    meta = trace_record.metadata

    if trace_record.iteration == 1 && global_i > 1
        @printf("[DataLoader] Sampled new data batch.\n")
    end

    @printf("i = %d \t L(𝛉) = %.6f \t Δt = %.2fs \t σ = %.5f\n", global_i, meta.L, elapsed, meta.σ)

    push!(
        cb.complete_trace, (
            i = global_i,
            L = Float32(meta.L),
            σ = Float32(meta.σ),
        )
    )

    if trace_record.iteration == cb.checkpoint_Δi
        θ_current = ComponentArray(meta.θ_ema, cb.axes)

        raw_acc = accuracy(cb.model, θ_current, cb.st, cb.test_dataloader)
        test_acc = raw_acc > 1.0 ? raw_acc / 100.0 : raw_acc

        push!(cb.accuracy_trace, (global_i, test_acc))

        if test_acc > cb.best_test_acc
            cb.best_test_acc = test_acc

            checkpoint_data = Dict(
                "i" => global_i,
                "L" => meta.L,
                "θ" => meta.θ,
                "σ" => meta.σ,
                "θ_ema" => meta.θ_ema,
                "test_acc" => test_acc,
                "complete_trace" => cb.complete_trace,
                "accuracy_trace" => cb.accuracy_trace
            )

            save_checkpoint(checkpoint_data, test_acc, global_i, cb.prev_checkpoint, cb.save_dir)

            @printf("\n[Checkpoint Saved] i = %d \t L(𝛉) = %.6f \t Accuracy = %.2f%%\n\n", global_i, meta.L, test_acc * 100.0)
        else
            @printf("\n[Skipped Checkpoint] i = %d \t L(𝛉) = %.6f \t Accuracy = %.2f%% (Best: %.2f%%)\n\n", global_i, meta.L, test_acc * 100.0, cb.best_test_acc * 100.0)
        end
    end
    return false
end

end
