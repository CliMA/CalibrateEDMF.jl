"""Initializes a SCM calibration process."""

using ArgParse
using CalibrateEDMF
using CalibrateEDMF.Pipeline

s = ArgParseSettings()
@add_arg_table s begin
    "--config"
    help = "Inverse problem config file"
    arg_type = String
    "--job_id"
    help = "Job identifier"
    arg_type = String
    default = "default_id"
end
parsed_args = parse_args(ARGS, s)
include(parsed_args["config"])

init_calibration(get_config(); job_id = parsed_args["job_id"])
