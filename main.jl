using Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Printf
using MLDatasets: MNIST
using SimpleChains: SimpleChains

function load_MNIST(; batch_size)
    train_data = MNIST(:train)
    test_data = MNIST(:test)

    x_train = Float32.(reshape(train_data.features, 28, 28, 1, :))
    y_train = onehotbatch(train_data.targets, 0:9)

    x_test = Float32.(reshape(test_data.features, 28, 28, 1, :))
    y_test = onehotbatch(test_data.targets, 0:9)

    return (
        DataLoader(
            (x_train, y_train);
            batchsize = batch_size, shuffle = true
        ),
        DataLoader(
            (x_test, y_test);
            batchsize = batch_size, shuffle = false
        ),
    )
end

function accuracy(model, θ, s, dataloader)
    s = Lux.testmode(s)
    correct = sum(dataloader) do (x, y)
        sum(onecold(y) .== onecold(Array(first(model(x, θ, s)))))
    end
    total = sum(size(y, 2) for (x, y) in dataloader)
    return (correct / total) * 100
end

function train(model; rng = Random.default_rng())
    train_dataloader, test_dataloader = load_MNIST(batch_size = 128)
    θ, s = Lux.setup(rng, model)
    ts = Training.TrainState(model, θ, s, Adam(3.0f-4))
    lossfn = CrossEntropyLoss(; logits = Val(true))
    vjp = AutoZygote()

    n_epochs = 50

    for epoch in 1:n_epochs
        stime = time()
        for (x, y) in train_dataloader
            _, _, _, ts = Training.single_train_step!(vjp, lossfn, (x, y), ts)
        end
        ttime = time() - stime

        if epoch == 1 || epoch % 10 == 0 || epoch == n_epochs
            train_acc = accuracy(model, ts.parameters, ts.states, train_dataloader)
            test_acc = accuracy(model, ts.parameters, ts.states, test_dataloader)
            @printf "[%2d/%2d] \t Time %.2fs \t Training Accuracy: %.2f%% \t Test Accuracy: %.2f%%\n" epoch n_epochs ttime train_acc test_acc
        end
    end

    return ts

end

model = Chain(
    Conv((3, 3), 1 => 8, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 8 => 16, relu),
    MaxPool((2, 2)),
    FlattenLayer(3),
    Dense(5 * 5 * 16 => 120, relu),
    Dense(120 => 10)
)

simple_chains_model = ToSimpleChainsAdaptor((28, 28, 1))(model)
trained_state = train(simple_chains_model)
