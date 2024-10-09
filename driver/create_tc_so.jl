using Glob

if isempty(Glob.glob("CEDMF.so"))

    using Pkg
    using PackageCompiler
    using CalibrateEDMF

    cedmf = pkgdir(CalibrateEDMF)
    pkgs = [:CalibrateEDMF]
    append!(pkgs, [Symbol(v.name) for v in values(Pkg.dependencies()) if v.is_direct_dep])

    # Avoid unnecessary pkgs, and packages that are in julia's Base.loaded_modules (this is probably out of date now that stdlib is moved out of base, and also probably since we're moving across architectures it's safer to include everything?)
    do_not_compile_pkgs = [
        # :CairoMakie,
        # :Makie,
        # :ForwardDiff,
        # :PackageCompiler,
        # :NPZ,
        # :Test,
        # :Dates,
        # :LinearAlgebra,
        # :Statistics,
        # :Random,
        # :Logging,
        # :SparseArrays,
        # :TerminalLoggers,
        # :OrdinaryDiffEq,
        # :StochasticDiffEq,
        # :DiffEqBase,
        # :SciMLBase,
    ]
    filter!(pkg -> pkg ∉ do_not_compile_pkgs, pkgs)

    # switch to creating separate sysimage for each architecture on caltech cluster and see if that works -- however, the script can call the sysimage for its specific architecture
    # need to create for all bc the caller doesn't know which architecture its children will receive 
    # the original problem we had was probably tied to the construction of packages in the depot initially...? or maybe julia was built w/ different target on different machines? then perhaps creating targets on different nodes individually will help...
   
    architectures = ["skylake-avx512", "broadwell", "haswell", "sandybridge"] # do we actually have all 4 of these? can all these be created from one architecture? or do we need to call from sysimage.sbatch differently for each architecture?
   
    # should we stop it from rerunning if the image is already there? I guess not bc we're making changes in CEDMF itself that need to be in the image...
    for cpu_target in architectures
        create_sysimage(
            pkgs;
            sysimage_path = "CEDMF_$cpu_target.so",
            # Caltech Central CPU architecture, `native` leads to issues as well.
            # This one works most of the time, but needs a failsafe mechanism.
            cpu_target = cpu_target,
            precompile_execution_file = joinpath(cedmf, "test", "runtests.jl"), # what does this do?
        )
    end

    # Other cpu_target options
    #cpu_target = "native", # Default 
    #cpu_target = PackageCompiler.default_app_cpu_target(),
    #cpu_target = "generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)", # Same as Julia Base
    #cpu_target = "generic",

end
