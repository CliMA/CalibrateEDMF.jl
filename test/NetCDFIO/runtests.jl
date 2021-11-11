using Test
using LinearAlgebra
using NCDatasets
const NC = NCDatasets
using CalibrateEDMF.ModelTypes
using CalibrateEDMF.ReferenceModels
using CalibrateEDMF.ReferenceStats
using CalibrateEDMF.NetCDFIO
using CalibrateEDMF.TurbulenceConvectionUtils

@testset "NetCDFIO_Diags" begin
    # Choose same SCM to speed computation
    data_dir = mktempdir()
    scm_dirs = repeat([joinpath(data_dir, "Output.Bomex.000000")], 2)
    kwargs_ref_model = Dict(
        :y_names => [["u_mean"], ["v_mean"]],
        :y_dir => scm_dirs,
        :scm_dir => scm_dirs,
        :case_name => repeat(["Bomex"], 2),
        :t_start => repeat([4.0 * 3600], 2),
        :t_end => repeat([6.0 * 3600], 2),
    )
    # Generate ref_stats
    ref_models = construct_reference_models(kwargs_ref_model)
    run_SCM(ref_models, overwrite = false)
    ref_stats = ReferenceStatistics(ref_models, true, true; y_type = SCM(), Σ_type = SCM())
    # Generate config
    config = Dict()
    config["process"] = Dict()
    config["process"]["N_ens"] = 10
    config["prior"] = Dict()
    config["prior"]["constraints"] = Dict("foo" => 1, "bar" => 2)

    diags = NetCDFIO_Diags(config, data_dir, ref_stats)

    # Test constructor
    @test isa(diags, NetCDFIO_Diags)
    @test isempty(diags.vars)

    # Test reference diagnostics
    io_reference(diags, ref_stats)
    NC.Dataset(diags.filepath, "r") do root_grp
        ref_grp = root_grp.group["reference"]
        @test ref_grp["Gamma"] ≈ ref_stats.Γ
        @test ref_grp["Gamma_full"] ≈ ref_stats.Γ_full
        @test ref_grp["y"] ≈ ref_stats.y
        @test ref_grp["y_full"] ≈ ref_stats.y_full
    end

    # Test iteration-dependent diagnostics
    open_files(diags)
    init_iteration_io(diags)
    write_iteration(diags)
    close_files(diags)
    NC.Dataset(diags.filepath, "r") do root_grp
        ensemble_grp = root_grp.group["ensemble_diags"]
        @test length(ensemble_grp["iteration"]) == 2
    end
end
