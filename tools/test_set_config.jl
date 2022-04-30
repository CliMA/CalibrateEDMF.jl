# Cases must include `case_name` and `scm_suffix`. If `case_name`=="LES_driven_SCM", must also include `y_dir`.
struct ScmTest end

function get_reference_config(::ScmTest)
    config = Dict()

    # amip4K data: April
    cfsite_numbers = (13, 21, 23)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 4, experiment = "amip4K")

    ref_dirs = [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers]
    suffixes = [get_gcm_les_uuid(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers]
    n_repeat = length(ref_dirs)
    config["case_name"] = repeat(["LES_driven_SCM"], n_repeat)
    config["y_dir"] = ref_dirs
    config["scm_suffix"] = suffixes
    return config
end
