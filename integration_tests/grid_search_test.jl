using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere begin
    using CalibrateEDMF
    ce = pkgdir(CalibrateEDMF)
    gs = joinpath(ce, "experiments", "grid_search")
    include(joinpath(gs, "grid_search.jl"))
    include(joinpath(gs, "loss_map.jl"))
    include(joinpath(gs, "quick_plots.jl"))
end
using Test
import JSON

config_path = joinpath(@__DIR__, "grid_search_integration_test_config.jl")
include(config_path)
config = get_config()  # 
out_dir = joinpath(@__DIR__, "output", "grid_search")
rm(out_dir, recursive = true, force = true)  # delete old test output
@time begin
    grid_search(config, config_path, out_dir)
    @info "Grid search complete"
end
y_root = joinpath(out_dir, "general_stochastic_ent_params_{1}.general_stochastic_ent_params_{2}/0.3_0.2")
y_dirs = [
    joinpath(y_root, "TRMM_LBA.1/Output.TRMM_LBA.1"),
    joinpath(y_root, "Rico.1/Output.Rico.1"),
    joinpath(y_root, "TRMM_LBA.2/Output.TRMM_LBA.1"),
]
nc_files = joinpath.(y_dirs, ["stats/Stats.TRMM_LBA.nc", "stats/Stats.Rico.nc", "stats/Stats.TRMM_LBA.nc"])
@test all(isfile.(nc_files))

# update reference data paths
config["reference"]["y_dir"] = y_dirs

@time begin
    compute_loss_map(config, out_dir)
    @info "Loss map complete"
end

quick_plots(joinpath(out_dir, "loss.nc"), out_dir)

# Additional testing:
namelist_files = joinpath.(y_dirs, ["namelist_TRMM_LBA.in", "namelist_Rico.in", "namelist_TRMM_LBA.in"])
nl1 = open(namelist_files[1], "r") do io
    JSON.parse(io; dicttype = Dict, inttype = Int64)
end
@test nl1["time_stepping"]["t_max"] == 199.0
nl2 = open(namelist_files[2], "r") do io
    JSON.parse(io; dicttype = Dict, inttype = Int64)
end
@test nl2["time_stepping"]["t_max"] == 200.0
nl3 = open(namelist_files[3], "r") do io
    JSON.parse(io; dicttype = Dict, inttype = Int64)
end
@test nl3["time_stepping"]["t_max"] == 201.0
