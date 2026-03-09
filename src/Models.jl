module Models

import SimpleChains
using Lux: Chain, Conv, MaxPool, FlattenLayer, Dense, relu, ToSimpleChainsAdaptor
using WeightInitializers: kaiming_normal

export CNN_2C2D_MNIST

const CNN_2C2D_MNIST = ToSimpleChainsAdaptor((28, 28, 1))(
    Chain(
        Conv((3, 3), 1 => 8, relu; init_weight = kaiming_normal),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu; init_weight = kaiming_normal),
        MaxPool((2, 2)),
        FlattenLayer(3),
        Dense(5 * 5 * 16 => 120, relu; init_weight = kaiming_normal),
        Dense(120 => 10; init_weight = kaiming_normal)
    )
)

end
