using CalibrateEDMF, Documenter
using DocumenterCitations

# https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846
ENV["GKSwstype"] = "nul"

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

#! format: off
api = [
    "Diagnostics" => "API/Diagnostics.md",
    "ReferenceModels" => "API/ReferenceModels.md",
    "ReferenceStats" => "API/ReferenceStats.md",
    "Pipeline" => "API/Pipeline.md",
    "TurbulenceConvectionUtils" => "API/TurbulenceConvectionUtils.md",
    "KalmanProcessUtils" => "API/KalmanProcessUtils.md",
    "DistributionUtils" => "API/DistributionUtils.md",
    "LESUtils" => "API/LESUtils.md",
    "HelperFuncs" => "API/HelperFuncs.md",
]

pages = Any[
    "Home" => "index.md",
    "Installation instructions" => "installation.md",
    "Getting started" => "getting_started.md",
    "API" => api,
    "Running TC.jl with optimal parameters" => "tc_runner.md",
    "Contributing" => "contributing.md",
    "References" => "References.md",
]

mathengine = MathJax(Dict(
    :TeX => Dict(
        :equationNumbers => Dict(:autoNumber => "AMS"),
        :Macros => Dict(),
    ),
))

format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    mathengine = mathengine,
    collapselevel = 1,
)
#! format: on

makedocs(
    bib,
    sitename = "CalibrateEDMF.jl",
    # strict = true, # TODO: make docs strict
    format = format,
    # checkdocs = :exports, # TODO: remove comment when doc strings are added to API
    clean = true,
    doctest = true,
    modules = [CalibrateEDMF],
    pages = pages,
)

deploydocs(
    repo = "github.com/CliMA/CalibrateEDMF.jl.git",
    target = "build",
    push_preview = true,
    devbranch = "main",
    forcepush = true,
)
