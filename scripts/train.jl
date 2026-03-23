Base.exit_on_sigint(false)
import Pkg; Pkg.activate(".")
using Kiseki, ArgParse
using Lux: gpu_device, cpu_device

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--resume", "-r"
        default = nothing

        "--optimizer", "-o"
        help = "leea | sgd"
        range_tester = x -> lowercase(x) in ["leea", "sgd"]
        default = "leea"

        "--device", "-d"
        help = "gpu | cpu"
        range_tester = x -> lowercase(x) in ["gpu", "cpu"]
        default = "gpu"

        "--seed", "-S"
        arg_type = Int
        default = 42

        "--batchsize", "-b"
        arg_type = Int
        default = 500

        "--iterations", "-i"
        arg_type = Int
        default = 100000

        "--target-acc", "-t"
        help = "[0.0, 100.0]"
        arg_type = Float64
        range_tester = x -> 0.0 <= x <= 100.0
        default = 100.0

        "--val-freq", "-v"
        arg_type = Int
        default = 10

        "--save-freq", "-s"
        arg_type = Int
        default = 50
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    opts = Dict("leea" => LEEA(), "sgd" => SGD())

    exp = Experiment(
        opt = opts[lowercase(args["optimizer"])],
        device = lowercase(args["device"]) == "cpu" ? cpu_device() : gpu_device(),
        seed = args["seed"],
        batchsize = args["batchsize"],
        max_i = args["iterations"],
        target_acc = args["target-acc"],
        val_freq = args["val-freq"],
        save_freq = args["save-freq"]
    )

    if !isnothing(args["resume"])
        est = load_checkpoint(args["resume"])
        run(exp, est)
    else
        run(exp)
    end

    return
end

function start()
    try
        main()
    catch e
        if e isa InterruptException
            exit(0)
        else
            rethrow(e)
        end
    end
    return
end

start()
