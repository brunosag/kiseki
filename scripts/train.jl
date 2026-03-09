Base.exit_on_sigint(false)
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Kiseki, ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--batchsize", "-b"
        arg_type = Int
        required = false
        default = 10

        "--seed", "-s"
        arg_type = Int
        required = false
        default = 42

        "--generations", "-g"
        arg_type = Int
        required = false
        default = 500000
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    batchsize = parsed_args["batchsize"]
    seed = parsed_args["seed"]
    generations = parsed_args["generations"]

    train_LEEA(; seed, batchsize, generations)

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
