using Kiseki
import Kiseki: on_step_end!, on_val_end!
using Lux: gpu_device, cpu_device
using GenieFramework
using StippleLatex
using PlotlyBase
@genietools

struct StippleCallback <: AbstractCallback
    update_fn::Function
end

on_step_end!(cb::StippleCallback, exp, est, loss, Δt) = cb.update_fn(est)
on_val_end!(cb::StippleCallback, exp, est, val_set, model, θ, st, acc, is_best) = cb.update_fn(est)

const OPTIMIZERS = Dict(
    "SGD" => [
        Dict("key" => "η", "label" => raw"\eta", "default" => 0.01, "type" => "number", "step" => 0.01, "desc" => "Learning rate")
    ],
    "LEEA" => [
        Dict("key" => "N", "label" => raw"N", "default" => 200, "type" => "number", "step" => 1, "desc" => "Population size"),
        Dict("key" => "pₘ", "label" => raw"p_{\mathrm{m}}", "default" => 0.04, "type" => "number", "step" => 0.01, "desc" => "Mutation probability"),
        Dict("key" => "η₀", "label" => raw"\eta_0", "default" => 0.03, "type" => "number", "step" => 0.01, "desc" => "Initial mutation step size"),
        Dict("key" => "γ", "label" => raw"\gamma", "default" => 0.99, "type" => "number", "step" => 0.01, "desc" => "Mutation decay factor"),
        Dict("key" => "ρ", "label" => raw"\rho", "default" => 0.4, "type" => "number", "step" => 0.01, "desc" => "Retention fraction"),
        Dict("key" => "ρₓ", "label" => raw"\rho_{\mathrm{x}}", "default" => 0.5, "type" => "number", "step" => 0.01, "desc" => "Crossover fraction"),
        Dict("key" => "λ", "label" => raw"\lambda", "default" => 0.2, "type" => "number", "step" => 0.01, "desc" => "Fitness decay coefficient"),
        Dict("key" => "τ_pat", "label" => raw"\tau_{\mathrm{pat}}", "default" => 25, "type" => "number", "step" => 1, "desc" => "Validation patience threshold")
    ]
)

const STOP_SIGNAL = Threads.Atomic{Bool}(false)
const EXPERIMENT_RUNNING = Threads.Atomic{Bool}(false)
const ACTIVE_EST = Ref{Any}(nothing)

function update_ui!(model, est)
    model.current_step[] = est.i
    model.best_acc[] = Float64(est.best_acc)
    if !isempty(est.history.loss)
        model.current_loss[] = Float64(est.history.loss[end])
    end

    loss_y = Float64.(est.history.loss)
    loss_x = collect(1:length(loss_y))
    acc_x = [a.i for a in est.history.acc]
    acc_y = [Float64(a.value) for a in est.history.acc]

    model.plot_data[] = [
        scatter(
            x=loss_x, y=loss_y,
            name="Loss",
            mode="lines",
            yaxis="y1",
            line=PlotlyBase.attr(color="#18181b", width=1.5) # zinc-900
        ),
        scatter(
            x=acc_x, y=acc_y,
            name="Accuracy",
            mode="lines",
            yaxis="y2",
            line=PlotlyBase.attr(color="#a1a1aa", width=1.5) # zinc-400
        )
    ]
end

@app begin
    @in dataset = "mnist"
    @in device = "cpu"
    @in seed = 42
    @in batchsize = 1000
    @in max_i = 100000
    @in target_acc = 100.0

    @out optimizers_schema = OPTIMIZERS
    @in optimizer = "LEEA"

    @in opt_params = Dict{String,Any}(
        k => Dict{String,Any}(p["key"] => p["default"] for p in v)
        for (k, v) in OPTIMIZERS
    )

    @out current_step = 0
    @out current_loss = 0.0
    @out best_acc = 0.0

    @out is_running = false
    @in start_experiment = false
    @in stop_experiment = false

    @out plot_data = PlotlyBase.AbstractTrace[]
    @out plot_layout = PlotlyBase.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PlotlyBase.attr(family="system-ui, -apple-system, sans-serif", color="#52525b"), # zinc-600
        margin=PlotlyBase.attr(l=50, r=50, t=10, b=40),
        xaxis=PlotlyBase.attr(
            title="Step",
            gridcolor="#f4f4f5", # zinc-100
            zerolinecolor="#f4f4f5",
            tickcolor="#e4e4e7" # zinc-200
        ),
        yaxis=PlotlyBase.attr(
            title="Loss",
            side="left",
            gridcolor="#f4f4f5",
            zerolinecolor="#f4f4f5",
            tickfont=PlotlyBase.attr(color="#18181b"), # zinc-900
            titlefont=PlotlyBase.attr(color="#18181b")
        ),
        yaxis2=PlotlyBase.attr(
            title="Accuracy (%)",
            overlaying="y",
            side="right",
            showgrid=false,
            tickfont=PlotlyBase.attr(color="#71717a"), # zinc-500
            titlefont=PlotlyBase.attr(color="#71717a")
        ),
        legend=PlotlyBase.attr(
            x=0.5,
            y=1.05,
            xanchor="center",
            yanchor="bottom",
            orientation="h",
            bgcolor="rgba(255,255,255,0)"
        ),
        hovermode="x unified",
        hoverlabel=PlotlyBase.attr(
            bgcolor="#ffffff",
            bordercolor="#e4e4e7",
            font_color="#18181b",
            font_family="system-ui, -apple-system, sans-serif"
        )
    )

    @onchange isready begin
        if isready
            is_running = EXPERIMENT_RUNNING[]
            if EXPERIMENT_RUNNING[] && ACTIVE_EST[] !== nothing
                @async begin
                    while EXPERIMENT_RUNNING[]
                        est_current = ACTIVE_EST[]
                        if est_current !== nothing
                            update_ui!(__model__, est_current)
                        end
                        sleep(1.0)
                    end
                    is_running = false
                    if ACTIVE_EST[] !== nothing
                        update_ui!(__model__, ACTIVE_EST[])
                    end
                end
            elseif ACTIVE_EST[] !== nothing
                update_ui!(__model__, ACTIVE_EST[])
            end
        end
    end

    @onchange stop_experiment begin
        if stop_experiment
            EXPERIMENT_RUNNING[] = false
            STOP_SIGNAL[] = true
            stop_experiment = false
            is_running = false
        end
    end

    @onchange start_experiment begin
        if start_experiment
            try
                is_running = true
                EXPERIMENT_RUNNING[] = true
                start_experiment = false
                STOP_SIGNAL[] = false

                dev = device == "gpu" ? gpu_device() : cpu_device()

                kwargs = Dict{Symbol,Any}()
                for p in optimizers_schema[optimizer]
                    k = p["key"]
                    val = opt_params[optimizer][k]
                    kwargs[Symbol(k)] = p["step"] == 1 ? round(Int, val) : Float64(val)
                end

                opt_instance = getproperty(Kiseki, Symbol(optimizer))(; kwargs...)

                exp = Experiment(
                    opt=opt_instance,
                    device=dev,
                    seed=seed,
                    batchsize=batchsize,
                    max_i=max_i,
                    target_acc=Float64(target_acc)
                )

                stipple_callback = StippleCallback((est) -> begin
                    if STOP_SIGNAL[]
                        throw(InterruptException())
                    end
                end)

                est = Kiseki.init(exp, (stipple_callback,))
                ACTIVE_EST[] = est

                errormonitor(
                    Threads.@spawn begin
                        try
                            run!(exp, est)
                        catch e
                            if e isa InterruptException
                                @info "Experiment stopped by user."
                            else
                                @error "Experiment execution failed" exception = (e, catch_backtrace())
                            end
                        finally
                            sleep(0.2)
                            EXPERIMENT_RUNNING[] = false
                        end
                    end
                )

                @async begin
                    while EXPERIMENT_RUNNING[]
                        est_current = ACTIVE_EST[]
                        if est_current !== nothing
                            update_ui!(__model__, est_current)
                        end
                        sleep(1.0)
                    end
                    is_running = false
                    if ACTIVE_EST[] !== nothing
                        update_ui!(__model__, ACTIVE_EST[])
                    end
                end
            catch e
                is_running = false
                EXPERIMENT_RUNNING[] = false
                @error "Experiment setup failed" exception = (e, catch_backtrace())
            end
        end
    end
end

@page("/", "src/ui.html")

Server.up(async=false)
