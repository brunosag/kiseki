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

data = jldopen(checkpoint_path, "r") do file
    Dict(k => file[k] for k in keys(file))
end

trace = data["complete_trace"]

iterations = [t.i for t in trace]
loss = [t.L for t in trace]
σ_vals = [t.σ for t in trace]

p1 = plot(
    iterations, loss,
    title = "Objective Loss & Test Accuracy",
    xlabel = "Iteration",
    ylabel = "Loss",
    color = :red,
    legend = :right,
    label = "Training Loss",
    yticks = 0:0.2:2.2,
)

if haskey(data, "accuracy_trace")
    acc_trace = data["accuracy_trace"]
    if !isempty(acc_trace)
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
            ylabel = "Test Accuracy (%)",
            legend = false,
            ylims = (0, 100),
            yticks = 0:10:100,
        )
    end
end

p2 = plot(
    iterations, σ_vals,
    title = "Strategy Parameter Dynamics",
    xlabel = "Iteration",
    ylabel = L"\sigma",
    color = :blue,
    legend = false,
    top_margin = 20Plots.mm,
)

fig = plot(p1, p2, layout = (2, 1), size = (1000, 1000), margin = 5Plots.mm)

display(fig)

println("Plot rendered. Press Enter to close the window.")
readline()
