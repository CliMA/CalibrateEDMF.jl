using ArgParse

include("postprocessing.jl")

function parse_commandline()

    s = ArgParseSettings()
    @add_arg_table s begin
        "--cedmf_output_dir"
        help = "Path to the results directory (CEDMF output)"
        arg_type = String
        "--tc_output_dir"
        help = "Path to the results postproccessing TC runs output"
        arg_type = String
        "--save_dir"
        help = "Path to store output"
        arg_type = String
        default = nothing
        "--overwrite"
        help = "if to overwrite existing output"
        arg_type = Bool
        default = false
        "--overwrite_reference"
        help = "if to overwrite existing reference output"
        arg_type = Bool
        default = false
        "--reference_files_already_have_derived_data_vars_LES"
        help = "if the reference files already have derived data vars"
        arg_type = Bool
        default = false
        "--delete_TC_output_files"
        help = "if to delete TC output files"
        arg_type = Bool
        default = false
    end

    return ArgParse.parse_args(s)
end


if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    # get config file
    CEDMF_output_dir = abspath(args["cedmf_output_dir"])
    tc_output_dir = abspath(args["tc_output_dir"])
    save_dir = abspath(args["save_dir"])
    overwrite = args["overwrite"]
    overwrite_reference = args["overwrite_reference"]
    reference_files_already_have_derived_data_vars_LES = args["reference_files_already_have_derived_data_vars_LES"]
    delete_TC_output_files = args["delete_TC_output_files"]

    collate_postprocess_runs(
        CEDMF_output_dir,  # path to directory w/ Diagnostics.nc and config files etc
        tc_output_dir,  # path to directory w/ Diagnostics.nc and config files etc
        save_dir, # path to directory to save postprocessed files
        ;
        overwrite = overwrite,
        overwrite_reference = overwrite_reference,
        reference_files_already_have_derived_data_vars_LES = reference_files_already_have_derived_data_vars_LES,
        delete_TC_output_files = delete_TC_output_files,
        )
end