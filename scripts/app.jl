using Kiseki, GenieFramework, StippleLatex, PlotlyBase
import DataStructures: OrderedDict
import Lux: gpu_device, cpu_device
@genietools

const CONFIG_SCHEMA = OrderedDict(
    "dataset" => Dict("type" => "select", "label" => "Dataset", "options" => [
        Dict("label" => "MNIST", "value" => "mnist"),
        Dict("label" => "Fashion MNIST", "value" => "fashion"),
        Dict("label" => "CIFAR-10", "value" => "cifar10"),
    ]),
    "device" => Dict("type" => "select", "label" => "Device", "options" => [
        Dict("label" => "CPU", "value" => "cpu"),
        Dict("label" => "GPU", "value" => "gpu"),
    ]),
    "seed" => Dict("type" => "number", "step" => 1, "default" => 42, "label" => "Seed"),
    "batch_size" => Dict("type" => "number", "step" => 1, "default" => 1000, "label" => "Batch size"),
    "iterations" => Dict("type" => "number", "step" => 1, "default" => 100000, "label" => "Iterations"),
    "target_acc" => Dict("type" => "number", "step" => 0.01, "default" => 100.0, "label" => "Target accuracy"),
    "optimizer" => Dict("type" => "select", "label" => "Optimizer", "options" => [
        Dict("label" => "LEEA", "value" => "LEEA"),
        Dict("label" => "SGD", "value" => "SGD")
    ])
)

const OPTIMIZERS_SCHEMA = OrderedDict(
    "SGD" => [
        Dict("key" => "η", "label" => raw"\eta", "type" => "number", "default" => 0.01, "step" => 0.01, "desc" => "Learning rate"),
    ],
    "LEEA" => [
        Dict("key" => "N", "label" => raw"N", "type" => "number", "default" => 200, "step" => 1, "desc" => "Population size"),
        Dict("key" => "pₘ", "label" => raw"p_{\mathrm{m}}", "type" => "number", "default" => 0.04, "step" => 0.01, "desc" => "Mutation probability"),
        Dict("key" => "η₀", "label" => raw"\eta_0", "type" => "number", "default" => 0.03, "step" => 0.01, "desc" => "Initial mutation step size"),
        Dict("key" => "γ", "label" => raw"\gamma", "type" => "number", "default" => 0.99, "step" => 0.01, "desc" => "Mutation decay factor"),
        Dict("key" => "ρ", "label" => raw"\rho", "type" => "number", "default" => 0.4, "step" => 0.01, "desc" => "Retention fraction"),
        Dict("key" => "ρₓ", "label" => raw"\rho_{\mathrm{x}}", "type" => "number", "default" => 0.5, "step" => 0.01, "desc" => "Crossover fraction"),
        Dict("key" => "λ", "label" => raw"\lambda", "type" => "number", "default" => 0.2, "step" => 0.01, "desc" => "Fitness decay coefficient"),
        Dict("key" => "τ_pat", "label" => raw"\tau_{\mathrm{pat}}", "type" => "number", "default" => 25, "step" => 1, "desc" => "Validation patience threshold"),
    ]
)

const PLOT_LAYOUT = PlotlyBase.Layout(
    xaxis=attr(title="Step", fixedrange=false, gridcolor="#f4f4f5"), # zinc-100
    yaxis=attr(title="Loss", fixedrange=false, gridcolor="#f4f4f5"), # zinc-100
    yaxis2=attr(title="Accuracy (%)", fixedrange=false, overlaying="y", side="right", showgrid=false),
    legend=attr(orientation="h", x=0.5, y=1, xanchor="center", yanchor="bottom"),
    font=attr(family="sans-serif"),
    paper_bgcolor="transparent",
    plot_bgcolor="transparent",
    dragmode=false,
)

const PLOT_CONFIG = PlotlyBase.PlotConfig(
    displayModeBar=false,
    displaylogo=false,
    scrollZoom=false,
)

@kwdef mutable struct ExperimentConfig
    dataset::String = CONFIG_SCHEMA["dataset"]["options"][1]["value"]
    device::String = CONFIG_SCHEMA["device"]["options"][1]["value"]
    seed::Int = CONFIG_SCHEMA["seed"]["default"]
    batch_size::Int = CONFIG_SCHEMA["batch_size"]["default"]
    iterations::Int = CONFIG_SCHEMA["iterations"]["default"]
    target_acc::Float64 = CONFIG_SCHEMA["target_acc"]["default"]
    optimizer::String = CONFIG_SCHEMA["optimizer"]["options"][1]["value"]
end

const stop_signal = Threads.Atomic{Bool}(false)
global est::Union{ExperimentState,Nothing} = nothing

@kwdef struct StippleCallback <: AbstractCallback
    update_step::Function
    update_val::Function
    last_t::Threads.Atomic{Float64}
    throttle_sec::Float64
end

function Kiseki.on_step_end!(cb::StippleCallback, exp, est, loss, Δt)
    if stop_signal[]
        Threads.atomic_xchg!(stop_signal, false)
        throw(InterruptException())
    end

    t = Base.time()
    if t - cb.last_t[] > cb.throttle_sec
        Threads.atomic_xchg!(cb.last_t, t)
        cb.update_step(est, loss, Δt)
    end
end

Kiseki.on_val_end!(cb::StippleCallback, exp, est, val_set, model, θ, st, acc, is_best) = cb.update_val(est, acc)

@app begin
    # Constants
    @out config_schema = CONFIG_SCHEMA
    @out optimizers_schema = OPTIMIZERS_SCHEMA
    @out plot_layout = PLOT_LAYOUT
    @out plot_config = PLOT_CONFIG

    # Input
    @in config = ExperimentConfig()
    @in opt_params = Dict{String,Any}(
        opt => Dict{String,Any}(param["key"] => param["default"] for param in params)
        for (opt, params) in pairs(OPTIMIZERS_SCHEMA)
    )

    # State
    @out is_running = !isnothing(est)
    @out current_step = 0
    @out best_acc = 0.0
    @out current_loss = 0.0
    @out plot_data = AbstractTrace[
        scatter(name="Loss", x=Int[], y=Float64[]),
        scatter(name="Accuracy", x=Int[], y=Float64[], yaxis="y2")
    ]

    # Actions
    @in start_experiment = false
    @onbutton start_experiment begin
        try
            @info "Starting experiment"
            Threads.atomic_xchg!(stop_signal, false)
            is_running = true

            opt_kwargs = NamedTuple(
                Symbol(param["key"]) => opt_params[config.optimizer][param["key"]]
                for param in optimizers_schema[config.optimizer]
            )
            exp = Experiment(
                device=config.device == "gpu" ? gpu_device() : cpu_device(),
                seed=config.seed,
                batchsize=config.batch_size,
                max_i=config.iterations,
                target_acc=config.target_acc,
                opt=getproperty(Kiseki, Symbol(config.optimizer))(; opt_kwargs...)
            )

            reactive_callback = StippleCallback(
                update_step=(est, loss, Δt) -> begin
                    __model__.current_step[] = est.i
                    __model__.current_loss[] = !isempty(est.history.loss) ? est.history.loss[end] : 0.0
                    __model__.plot_data[][1] = scatter(
                        name="Loss",
                        x=eachindex(est.history.loss),
                        y=est.history.loss,
                        line=PlotlyBase.attr(color="#18181b", width=1.5) # zinc-900
                    )
                end,
                update_val=(est, acc) -> begin
                    __model__.best_acc[] = est.best_acc
                    __model__.plot_data[][2] = scatter(
                        name="Accuracy",
                        x=[a.i for a in est.history.acc],
                        y=[a.value for a in est.history.acc],
                        yaxis="y2",
                        line=PlotlyBase.attr(color="#a1a1aa", width=1.5) # zinc-400
                    )
                end,
                last_t=Threads.Atomic{Float64}(Base.time()),
                throttle_sec=0.1
            )

            global est = Kiseki.init(exp, (reactive_callback,))

            Threads.@spawn run!(exp, est)
        catch e
            @error "Exception caught in start_experiment:" exception = (e, catch_backtrace())
            is_running = false
        end
    end

    @in stop_experiment = false
    @onbutton stop_experiment begin
        @info "Stopping experiment"
        Threads.atomic_xchg!(stop_signal, true)
        global est = nothing
        is_running = false
    end
end

@page("/", joinpath(@__DIR__, "ui.html"))

Server.up(async=true)
