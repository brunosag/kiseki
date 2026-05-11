using Kiseki, GenieFramework, StippleLatex
import DataStructures: OrderedDict
import Lux: gpu_device, cpu_device
@genietools

const CONFIG_SCHEMA = OrderedDict(
    "dataset" => Dict("type" => "select", "label" => "Dataset", "options" => [
        Dict("label" => "MNIST", "value" => "mnist"),
        Dict("label" => "Fashion MNIST", "value" => "fashion"),
        Dict("label" => "CIFAR-10", "value" => "cifar10")
    ]),
    "device" => Dict("type" => "select", "label" => "Device", "options" => [
        Dict("label" => "CPU", "value" => "cpu"),
        Dict("label" => "GPU", "value" => "gpu")
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
        Dict("key" => "η", "label" => raw"\eta", "type" => "number", "default" => 0.01, "step" => 0.01, "desc" => "Learning rate")
    ],
    "LEEA" => [
        Dict("key" => "N", "label" => raw"N", "type" => "number", "default" => 200, "step" => 1, "desc" => "Population size"),
        Dict("key" => "pₘ", "label" => raw"p_{\mathrm{m}}", "type" => "number", "default" => 0.04, "step" => 0.01, "desc" => "Mutation probability"),
        Dict("key" => "η₀", "label" => raw"\eta_0", "type" => "number", "default" => 0.03, "step" => 0.01, "desc" => "Initial mutation step size"),
        Dict("key" => "γ", "label" => raw"\gamma", "type" => "number", "default" => 0.99, "step" => 0.01, "desc" => "Mutation decay factor"),
        Dict("key" => "ρ", "label" => raw"\rho", "type" => "number", "default" => 0.4, "step" => 0.01, "desc" => "Retention fraction"),
        Dict("key" => "ρₓ", "label" => raw"\rho_{\mathrm{x}}", "type" => "number", "default" => 0.5, "step" => 0.01, "desc" => "Crossover fraction"),
        Dict("key" => "λ", "label" => raw"\lambda", "type" => "number", "default" => 0.2, "step" => 0.01, "desc" => "Fitness decay coefficient"),
        Dict("key" => "τ_pat", "label" => raw"\tau_{\mathrm{pat}}", "type" => "number", "default" => 25, "step" => 1, "desc" => "Validation patience threshold")
    ]
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

struct StippleCallback <: AbstractCallback end
function Kiseki.on_step_end!(cb::StippleCallback, exp, est, loss, Δt)
    if stop_signal[]
        Threads.atomic_xchg!(stop_signal, false)
        throw(InterruptException())
    end
end

@app begin
    # Constants
    @out config_schema = CONFIG_SCHEMA
    @out optimizers_schema = OPTIMIZERS_SCHEMA

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
    @out loss_history = []
    @out acc_history = []

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

            global est = Kiseki.init(exp, (StippleCallback(),))

            Threads.@spawn run!(exp, est)

            errormonitor(Threads.@spawn begin
                while is_running
                    current_step = est.i
                    best_acc = est.best_acc
                    loss_history = est.history.loss
                    acc_history = est.history.acc
                    sleep(0.5)
                end
            end)
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
