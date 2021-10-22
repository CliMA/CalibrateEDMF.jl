using CalibrateEDMF, Documenter
using DocumenterCitations

# https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846
ENV["GKSwstype"] = "nul"

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

#! format: off
pages = Any[
    "Home" => "index.md",
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
