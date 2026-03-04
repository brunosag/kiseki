using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Kiseki, Random, ArgParse


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "mode"
        help = "Specify the optimizer: 'evolution' or 'gradient'"
        required = true
        range_tester = (x -> x in ["evolution", "gradient"])
        "resume"
        help = "Path to a .jld2 checkpoint file to resume training"
        required = false
    end

    return parse_args(s)
end


function main()
    parsed_args = parse_commandline()
    mode = parsed_args["mode"]
    resume_target = parsed_args["resume"]

    rng = Xoshiro(42)

    checkpoint_dir = abspath(joinpath(@__DIR__, "..", "checkpoints"))
    mkpath(checkpoint_dir)

    model = create_mnist_model()

    if mode == "evolution"
        return train_evolution(
            model;
            rng = rng,
            resume_file = resume_target,
            save_dir = checkpoint_dir,
        )
    elseif mode == "gradient"
        config = GradientConfig(α = 3.0f-4, epochs = 200, batchsize = 128)
        return train_gradient(
            model;
            config = config,
            rng = rng,
            checkpoint_Δi = 100,
            resume_file = resume_target,
            save_dir = checkpoint_dir,
        )
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
