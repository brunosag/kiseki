using JLD2
using Plots
using LaTeXStrings

if length(ARGS) != 1
    println(stderr, "Usage: julia visualize.jl <path_to_checkpoint.jld2>")
    exit(1)
end

checkpoint_path = ARGS[1]

if !isfile(checkpoint_path)
    println(stderr, "Error: File not found at $checkpoint_path")
    exit(1)
end

trace, acc_trace = jldopen(checkpoint_path, "r") do file
    t = file["complete_trace"]
    a = haskey(file, "accuracy_trace") ? file["accuracy_trace"] : nothing
    (t, a)
end

iterations = [t.i for t in trace]
loss = [t.L for t in trace]
σ_vals = [t.σ for t in trace]

p1 = plot(
    iterations, loss,
    xlabel = "Iteration",
    ylabel = "Loss",
    color = :red,
    legend = :right,
    label = "Training Loss",
    right_margin = 10Plots.mm,
    top_margin = 5Plots.mm,
)

if !isnothing(acc_trace) && !isempty(acc_trace)
    acc_iters = [t[1] for t in acc_trace]
    acc_vals = [t[2] <= 1.0 ? t[2] * 100.0 : t[2] for t in acc_trace]

    plot!(
        p1, [NaN], [NaN],
        color = :green,
        label = "Test Accuracy",
    )

    p1_acc = twinx(p1)
    plot!(
        p1_acc, acc_iters, acc_vals,
        color = :green,
        ylabel = "Accuracy (%)",
        legend = false,
        ylims = (0, 100),
    )
end

p2 = plot(
    iterations, σ_vals,
    xlabel = "Iteration",
    ylabel = L"\sigma",
    color = :blue,
    legend = false,
)

fig = plot(p1, p2, layout = (2, 1), size = (1000, 1000), margin = 5Plots.mm)

display(fig)

println("Plot rendered. Press Enter to close the window.")
readline()
