module models

using Lux: Chain, Conv, MaxPool, FlattenLayer, Dense, relu
using WeightInitializers: kaiming_normal

export CNN_2C2D_MNIST

const CNN_2C2D_MNIST = Chain(
    Conv((3, 3), 1 => 8, relu; init_weight = kaiming_normal),
    MaxPool((2, 2)),
    Conv((3, 3), 8 => 16, relu; init_weight = kaiming_normal),
    MaxPool((2, 2)),
    FlattenLayer(3),
    Dense(5 * 5 * 16 => 120, relu; init_weight = kaiming_normal),
    Dense(120 => 10; init_weight = kaiming_normal)
)

end
