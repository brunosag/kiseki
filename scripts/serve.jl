using Kiseki, Oxygen, HTTP, JSON3, StructTypes
using Lux: gpu_device, cpu_device

const ACTIVE_CLIENTS = Set{WebSockets.WebSocket}()
const RUNNING_STATE = Ref{Union{Nothing, Tuple{Experiment, ExperimentState}}}(nothing)

Base.@kwdef struct ExperimentConfig
    dataset::String
    device::String
    seed::Int
    batchsize::Int
    max_i::Int
    target_acc::Float64
    opt::AbstractOptimizer
end
StructTypes.StructType(::Type{ExperimentConfig}) = StructTypes.Struct()

function parse_experiment_config(payload)
    config = JSON3.read(payload, ExperimentConfig)
    dev = config.device == "gpu" ? gpu_device() : cpu_device()

    return Experiment(
        opt = config.opt,
        device = dev,
        seed = config.seed,
        batchsize = config.batchsize,
        max_i = config.max_i,
        target_acc = config.target_acc
    )
end

@websocket "/experiment" function (ws)
    push!(ACTIVE_CLIENTS, ws)

    if !isnothing(RUNNING_STATE[])
        _, est = RUNNING_STATE[]
        msg = JSON3.write(
            (
                type = "sync",
                payload = (i = est.i, bestAcc = est.best_acc, history = (loss = est.history.loss, acc = est.history.acc)),
            )
        )
        WebSockets.send(ws, msg)
    end

    try
        for msg in ws
            exp = parse_experiment_config(msg)
            ws_logger = WebSocketLogger(ACTIVE_CLIENTS)
            est = init(exp, (ws_logger,))

            RUNNING_STATE[] = (exp, est)

            errormonitor(
                Threads.@spawn begin
                    run!(exp, est)
                end
            )
        end
    finally
        pop!(ACTIVE_CLIENTS, ws)
    end
end

@get "/checkpoints" function (req)
    checkpoints_dir = "checkpoints"

    if !isdir(checkpoints_dir)
        return []
    end

    files = readdir(checkpoints_dir)
    json_files = filter(f -> endswith(f, ".json"), files)

    checkpoints = map(json_files) do file
        filepath = joinpath(checkpoints_dir, file)
        file_content = read(filepath, String)
        return JSON3.read(file_content)
    end

    sort!(checkpoints, by = x -> x.timestamp, rev = true)

    return checkpoints
end

serve()
