abstract type AbstractOptimizer end
StructTypes.StructType(::Type{AbstractOptimizer}) = StructTypes.AbstractType()
StructTypes.subtypekey(::Type{AbstractOptimizer}) = :type
StructTypes.subtypes(::Type{AbstractOptimizer}) = (leea=LEEA, sgd=SGD)

abstract type AbstractOptimizerState end

init_workspace(opt::AbstractOptimizer, ops) = nothing
update_scheduler!(opt::AbstractOptimizer, ops, is_best) = nothing
format_metrics(ops::AbstractOptimizerState) = ""
