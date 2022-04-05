using Distributed
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    using CalibrateEDMF
    ce = pkgdir(CalibrateEDMF)
    include(joinpath(ce, "experiments", "grid_search", "grid_search.jl"))
    include(joinpath(ce, "experiments", "grid_search", "loss_map.jl"))
end
using Test

config_path = "grid_search_integration_test_config.jl"
include(config_path)
config = get_config()  # 
out_dir = joinpath(pwd(), "output", "grid_search")
# rm(out_dir, recursive = true, force = true)  # delete old test output
grid_search(config, config_path, "reference", out_dir)

y_root = joinpath(
    out_dir, 
    "general_stochastic_ent_params_{1}.general_stochastic_ent_params_{2}/0.4_0.35"
)
y_dirs = [
    joinpath(y_root, "TRMM_LBA.1/Output.TRMM_LBA.1/stats/Stats.TRMM_LBA.nc"),
    joinpath(y_root, "Rico.1/Output.Rico.1/stats/Stats.Rico.nc"),
    joinpath(y_root, "TRMM_LBA.2/Output.TRMM_LBA.1/stats/Stats.TRMM_LBA.nc"),
]
@test all(isfile.(y_dirs))

# update reference data paths
config["reference"]["y_dir"] = y_dirs

compute_loss_map(config, out_dir, "reference")
