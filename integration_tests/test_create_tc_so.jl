using Glob

if isempty(Glob.glob("CEDMF.so"))

    using Pkg
    using PackageCompiler
    using CalibrateEDMF

    cedmf = pkgdir(CalibrateEDMF)
    pkgs = [:CalibrateEDMF]
    append!(pkgs, [Symbol(v.name) for v in values(Pkg.dependencies()) if v.is_direct_dep])

    # Avoid unnecessary pkgs, and packages that are in julia's Base.loaded_modules
    do_not_compile_pkgs = [
        :CairoMakie,
        :PackageCompiler,
        :NPZ,
        :Test,
        :Dates,
        :LinearAlgebra,
        :Statistics,
        :Random,
        :Logging,
        :SparseArrays,
    ]
    filter!(pkg -> pkg ∉ do_not_compile_pkgs, pkgs)

    create_sysimage(
        pkgs;
        sysimage_path = "CEDMF.so",
        # Caltech Central CPU architecture, `native` leads to issues as well.
        # This one works most of the time, but needs a failsafe mechanism.
        cpu_target = "skylake-avx512",
        # precompile_execution_file = joinpath(cedmf, "test", "runtests.jl"),
    )

    # Other cpu_target options
    #cpu_target = "native", # Default 
    #cpu_target = PackageCompiler.default_app_cpu_target(),
    #cpu_target = "generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)", # Same as Julia Base
    #cpu_target = "generic",

end
