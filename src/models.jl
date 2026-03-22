adapt_model(model, device::AbstractCPUDevice) = Lux.ToSimpleChainsAdaptor((28, 28, 1))(model)
adapt_model(model, device::AbstractGPUDevice) = model

const CNN_2C2D_MNIST = Chain(
    conv1 = Conv((3, 3), 1 => 8, relu; init_weight = kaiming_normal),
    pool1 = MaxPool((2, 2)),
    conv2 = Conv((3, 3), 8 => 16, relu; init_weight = kaiming_normal),
    pool2 = MaxPool((2, 2)),
    flatten = FlattenLayer(3),
    dense1 = Dense(5 * 5 * 16 => 120, relu; init_weight = kaiming_normal),
    dense2 = Dense(120 => 10; init_weight = kaiming_normal)
)
