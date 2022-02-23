# include("experiments/scm_pycles_pipeline/julia_parallel_experimental/calibrate.jl")

# This is an example on training the TurbulenceConvection.jl implementation
# of the EDMF scheme with data generated using PyCLES (an LES solver) or
# TurbulenceConvection.jl (perfect model setting).
#
# This is ana experimental parallel script.

# Import modules to all processes
using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(dirname(dirname(dirname(@__DIR__))))
using Test

# Launch runners:
@everywhere begin
    include(joinpath(@__DIR__, "TCRunner.jl"))
end

# # Calibration process
N_iter = 5
@info "Running EK updates for $N_iter iterations"

function perform_ek_update()
    @info "Performing EK update..."
    sleep(1)
end
function print_estimated_cost(N_iter, p_workload)
    tc_first_eval = 200
    tc_compiled_eval = 20
    @assert N_iter > 0
    @assert p_workload ≥ 0
    n_sec = tc_first_eval + tc_compiled_eval * (p_workload - 1) * (N_iter - 1)
    n_min = n_sec / 60
    @info "Runtime cost estimate: $n_min minutes"
    return nothing
end

p_workload = 3
print_estimated_cost(N_iter, p_workload)

for iteration in 1:N_iter
    @time begin
        @info "Running iteration $iteration"
        pmap(TCmain, ntuple(i -> i, p_workload * (length(workers()))))
    end
    perform_ek_update()
end

@info "Done 🎉"
