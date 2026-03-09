module Kiseki

include("Data.jl")
include("Evaluation.jl")
include("Models.jl")
include("LEEA.jl")

using .Data
using .Evaluation
using .Models
using .LEEA

export load_MNIST, accuracy, CNN_2C2D_MNIST, train_LEEA

end
