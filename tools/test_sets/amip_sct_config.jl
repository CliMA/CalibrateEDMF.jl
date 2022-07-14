using CalibrateEDMF
import CalibrateEDMF.LESUtils: get_shallow_LES_library

# Get all shallow AMIP cases from Shen et al (2022)
les_library = get_shallow_LES_library()

struct ScmTest end
struct Amip_SCT end
namelist_args = [
    ("time_stepping", "dt_min", 0.5),
    ("time_stepping", "dt_max", 2.0),
    ("stats_io", "frequency", 60.0),
    ("grid", "stretch", "flag", true),
]

get_reference_config(::ScmTest) = get_reference_config(Amip_SCT())

# Cases must include `case_name`. If `case_name`=="LES_driven_SCM", must also include `y_dir`.
function get_reference_config(::Amip_SCT)
    config = Dict()

    ref_dirs = []
    for model in keys(les_library)
        for month in keys(les_library[model])
            cfsite_numbers = Tuple(les_library[model][month]["cfsite_numbers"])
            les_kwargs = (forcing_model = model, month = parse(Int, month), experiment = "amip")
            append!(ref_dirs, [get_cfsite_les_dir(cfsite_number; les_kwargs...) for cfsite_number in cfsite_numbers])
        end
    end
    n_repeat = length(ref_dirs)
    config["case_name"] = repeat(["LES_driven_SCM"], n_repeat)
    config["namelist_args"] = repeat([namelist_args], n_repeat)
    config["y_dir"] = ref_dirs
    return config
end
