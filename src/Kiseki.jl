module Kiseki

import Lux, Zygote, SimpleChains, Dates, Optimisers, JSON3
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
using Random: AbstractRNG, Xoshiro, TaskLocalRNG, rand!, shuffle, shuffle!
using MLDataDevices: AbstractDevice, AbstractCPUDevice, AbstractGPUDevice
using Lux: Chain, Conv, MaxPool, FlattenLayer, Dense, relu, logsoftmax, cpu_device, gpu_device

export Experiment, CNN_2C2D_MNIST, LEEA, SGD, run, load_checkpoint
export Tracker, CheckpointSaver, ConsoleLogger

include("data.jl")
include("models.jl")
include("callbacks.jl")
include("optimizers.jl")
include("optimizers/leea.jl")
include("optimizers/sgd.jl")
include("experiment.jl")

end
