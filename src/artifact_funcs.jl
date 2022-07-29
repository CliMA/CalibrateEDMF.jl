import ArtifactWrappers
const AW = ArtifactWrappers

"""
    get_path_to_artifact(casename = "Bomex", artifact_type = "PyCLES_output", artifact_dir = @__DIR__)

Downloads an artifact and returns its path.

# Arguments
- `casename`        :: Casename of the output to be downloaded, must be defined in `local_to_box` dict.
- `artifact_type`   :: Overarching type of artifact to be downloaded.
- `artifact_dir`    :: Location of the `Artifact.toml`.

# Returns
- The artifact path
"""
function get_path_to_artifact(casename = "Bomex", artifact_type = "PyCLES_output", artifact_dir = @__DIR__)
    local_to_box = Dict(
        "Bomex" => "https://caltech.box.com/shared/static/d6oo7th33839qmp4z99n8z4ryk3iepoq.nc",
        "TRMM_LBA" => "https://caltech.box.com/shared/static/wevi0rqiwo6sgkqdhcddr72u5ylt0tqp.nc",
    )
    lazy_download = true
    if haskey(local_to_box, casename)
        output_artifact = AW.ArtifactWrapper(
            artifact_dir,
            lazy_download,
            artifact_type,
            AW.ArtifactFile[AW.ArtifactFile(url = local_to_box[casename], filename = string(casename, ".nc")),],
        )
        output_artifact_path = AW.get_data_folder(output_artifact)
        @info "Artifact output path: $output_artifact_path"
        return output_artifact_path
    else
        throw(KeyError("Artifacts for casename $casename of type $artifact_type are not currently available."))
    end
end
