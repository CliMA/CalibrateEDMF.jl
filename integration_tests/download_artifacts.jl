include(joinpath(@__DIR__, "..", "src", "artifact_funcs.jl"))

function trigger_download()
    output_artifact_path = get_path_to_artifact()
    output_artifact_path2 = get_path_to_artifact("TRMM_LBA")
    return nothing
end
trigger_download()
