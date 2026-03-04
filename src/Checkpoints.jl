module Checkpoints

using JLD2
using Printf
using Dates

export load_checkpoint, save_checkpoint


function load_checkpoint(resume_file::String)
    if isfile(resume_file)
        @printf("Resuming from checkpoint: %s\n", resume_file)
        return jldopen(resume_file, "r") do file
            Dict{String, Any}(k => file[k] for k in keys(file))
        end
    end
    return nothing
end


function save_checkpoint(
        data::Dict, test_acc::Float64, global_iter::Int,
        prev_checkpoint::Ref{String}, dir::String, prefix::String
    )
    time_str = Dates.format(Dates.now(), "yyyy-mm-ddTHHMMSS")

    base_name = @sprintf("%s_%04dA_%dI_%s.jld2", prefix, round(Int, test_acc * 10000.0), global_iter, time_str)
    filename = joinpath(abspath(dir), base_name)
    temp_filename = filename * ".tmp"

    jldopen(temp_filename, "w") do file
        for (k, v) in data
            file[k] = v
        end
    end

    old_checkpoint = prev_checkpoint[]
    prev_checkpoint[] = filename

    mv(temp_filename, filename; force = true)

    if !isempty(old_checkpoint) && isfile(old_checkpoint)
        rm(old_checkpoint)
    end

    return filename
end


end
