module Kiseki

import Lux, Zygote
import Base: run
using LuxCUDA
using Printf
using Random: AbstractRNG, Xoshiro, TaskLocalRNG, rand!, shuffle, shuffle!
using Optimisers: destructure, Descent, Restructure
using OneHotArrays: onehotbatch, onecold
using ADTypes: AutoZygote
using StatsBase: Weights, sample, sample!
using MLDatasets: MNIST
using MLUtils: DataLoader, getobs, batch
using Lux: Chain, Conv, MaxPool, FlattenLayer, Dense, relu
using WeightInitializers: kaiming_normal
using MLDataDevices: AbstractDevice
using Polyester: @batch

export load_MNIST, CNN_2C2D_MNIST, AbstractOptimizer, AbstractOptimizerState
export LEEA, LEEAState, SGD, SGDState
export init, step!
export Experiment, ExperimentState, run

include("data.jl")
include("models.jl")
include("optimizers.jl")
include("optimizers/leea.jl")
include("optimizers/sgd.jl")
include("experiment.jl")

end
