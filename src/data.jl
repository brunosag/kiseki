module data

using MLDatasets: MNIST
using MLUtils: DataLoader, getobs, batch
using OneHotArrays: onehotbatch
using Random: AbstractRNG, shuffle, shuffle!
using StatsBase: sample

export load_MNIST

struct BalancedDataLoader{T <: AbstractArray, V <: AbstractVector, R <: AbstractRNG}
    X::T
    y::V
    batchsize::Int
    class_groups::Vector{Vector{Int}}
    samples_per_class::Int
    n_batches::Int
    rng::R

    function BalancedDataLoader(X::T, y::V, batchsize::Int, rng::R) where {T <: AbstractArray, V <: AbstractVector, R <: AbstractRNG}
        classes = sort(unique(y))
        n_classes = length(classes)

        if batchsize % n_classes != 0
            throw(ArgumentError("Batch size must be a multiple of $n_classes"))
        end

        samples_per_class = batchsize ÷ n_classes
        class_groups = [findall(==(c), y) for c in classes]
        n_batches = minimum(length.(class_groups)) ÷ samples_per_class

        return new{T, V, R}(X, y, batchsize, class_groups, samples_per_class, n_batches, rng)
    end
end

Base.length(d::BalancedDataLoader) = d.n_batches

function Base.iterate(d::BalancedDataLoader)
    shuffled_groups = [shuffle(d.rng, group) for group in d.class_groups]
    return iterate(d, (shuffled_groups, 1))
end

function Base.iterate(d::BalancedDataLoader, state)
    shuffled_groups, b = state
    if b > d.n_batches
        return nothing
    end

    batch_idx = Int[]
    sizehint!(batch_idx, d.batchsize)

    for group in shuffled_groups
        start_idx = (b - 1) * d.samples_per_class + 1
        end_idx = b * d.samples_per_class
        append!(batch_idx, @view group[start_idx:end_idx])
    end

    shuffle!(d.rng, batch_idx)

    X_raw, y_raw = getobs((d.X, d.y), batch_idx)

    Xᵢ = reshape(X_raw, 28, 28, 1, d.batchsize)
    Yᵢ = onehotbatch(y_raw, 0:9)

    return ((Xᵢ, Yᵢ), (shuffled_groups, b + 1))
end

function collate(batch_data)
    images = [obs[1] for obs in batch_data]
    labels = [obs[2] for obs in batch_data]

    Xᵢ = reshape(batch(images), 28, 28, 1, :)
    Yᵢ = onehotbatch(labels, 0:9)

    return (Xᵢ, Yᵢ)
end

function load_MNIST(rng::AbstractRNG, batchsize::Int; balanced::Bool = true)
    train_data = MNIST(:train)
    test_data = MNIST(:test)

    X_train, y_train = train_data.features, train_data.targets
    X_test, y_test = test_data.features, test_data.targets

    train_loader = balanced ?
        BalancedDataLoader(X_train, y_train, batchsize, rng) :
        DataLoader((X_train, y_train); batchsize, rng, collate, shuffle = true)
    test_loader = DataLoader((X_test, y_test); batchsize, collate)

    stateful_train_loader = Iterators.Stateful(Iterators.cycle(train_loader))

    return (stateful_train_loader, test_loader)
end

end
