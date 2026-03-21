module Kiseki

import Lux, Zygote, SimpleChains, Dates
import Base: run
using LuxCUDA, Printf
using Statistics: mean
using MLDatasets: MNIST
using ADTypes: AutoZygote
using WeightInitializers: kaiming_normal
using MLUtils: DataLoader, getobs, batch
using OneHotArrays: onehotbatch, onecold
using StatsBase: Weights, sample, sample!
using Serialization: serialize, deserialize
using Optimisers: destructure, Descent, Restructure
using Lux: Chain, Conv, MaxPool, FlattenLayer, Dense, relu, logsoftmax
using Random: AbstractRNG, Xoshiro, TaskLocalRNG, rand!, shuffle, shuffle!
using MLDataDevices: AbstractDevice, AbstractCPUDevice, AbstractGPUDevice

export Experiment, CNN_2C2D_MNIST, LEEA, SGD, run, load_checkpoint

include("data.jl")
include("models.jl")
include("optimizers.jl")
include("optimizers/leea.jl")
include("optimizers/sgd.jl")
include("experiment.jl")

end
