using Oxygen, HTTP, JSON3

@get "/api/checkpoints" function (req)
    checkpoints_dir = "checkpoints"

    if !isdir(checkpoints_dir)
        return []
    end

    files = readdir(checkpoints_dir)
    json_files = filter(f -> endswith(f, ".json"), files)

    checkpoints = map(json_files) do file
        filepath = joinpath(checkpoints_dir, file)
        file_content = read(filepath, String)
        return JSON3.read(file_content)
    end

    sort!(checkpoints, by = x -> x.timestamp, rev = true)

    return checkpoints
end

serve()
