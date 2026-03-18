abstract type AbstractOptimizer end
abstract type AbstractOptimizerState end

update_scheduler!(
    opt::AbstractOptimizer, ops::AbstractOptimizerState, acc, best_acc
) = nothing
