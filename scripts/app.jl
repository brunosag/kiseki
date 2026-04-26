using Kiseki
import Kiseki: on_step_end!, on_val_end!
using Lux: gpu_device, cpu_device
using GenieFramework
using StippleLatex
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

@app begin
    @in dataset = "mnist"
    @in device = "gpu"
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

    stop_signal = Threads.Atomic{Bool}(false)

    @onchange stop_experiment begin
        if stop_experiment
            is_running = false
            stop_signal[] = true
            stop_experiment = false
        end
    end

    @onchange start_experiment begin
        if start_experiment
            try
                is_running = true
                start_experiment = false
                stop_signal[] = false

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
                    if stop_signal[]
                        throw(InterruptException())
                    end

                    current_step = est.i
                    best_acc = Float64(est.best_acc)
                    if !isempty(est.history.loss)
                        current_loss = Float64(est.history.loss[end])
                    end
                end)

                est = Kiseki.init(exp, (stipple_callback,))

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
                            is_running = false
                        end
                    end
                )
            catch e
                is_running = false
                @error "Experiment setup failed" exception = (e, catch_backtrace())
            end
        end
    end
end

@page("/", "src/ui.html")

route("/checkpoints") do
    checkpoints_dir = "checkpoints"
    isdir(checkpoints_dir) || return Genie.Renderer.Json.json([])

    files = readdir(checkpoints_dir)
    json_files = filter(f -> endswith(f, ".json"), files)

    checkpoints = map(json_files) do file
        JSON3.read(read(joinpath(checkpoints_dir, file), String))
    end
    sort!(checkpoints, by=x -> x.timestamp, rev=true)

    return Genie.Renderer.Json.json(checkpoints)
end

Server.up(async=false)
