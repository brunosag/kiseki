Base.exit_on_sigint(false)
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Kiseki, ArgParse, LuxCUDA

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--batchsize", "-b"
        arg_type = Int
        required = false
        default = 500

        "--seed", "-s"
        arg_type = Int
        required = false
        default = 42

        "--iterations", "-i"
        arg_type = Int
        required = false
        default = 5
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    exp = Experiment(
        seed = parsed_args["seed"],
        batchsize = parsed_args["batchsize"],
        max_i = parsed_args["iterations"]
    )

    Kiseki.run(exp)

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
