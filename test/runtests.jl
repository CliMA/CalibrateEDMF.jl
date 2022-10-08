testlist = [
    "HelperFuncs",
    "DistributionUtils",
    "LESUtils",
    "KalmanProcessUtils",
    "TurbulenceConvectionUtils",
    "ReferenceModels",
    "ReferenceStats",
    "Pipeline",
    "NetCDFIO",
]

for test in testlist
    @time begin
        println("$(test):")
        include(joinpath(test, "runtests.jl"))
    end
end
