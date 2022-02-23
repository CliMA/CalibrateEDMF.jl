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
    myid() ≠ 1 && include(joinpath(@__DIR__, "TCRunner.jl"))
end

function signal_start_run(watch_file)
    open(watch_file, "w") do io
        println(io, "params_updated = true")
        println(io, "terminate = false")
        println(io, "solve_finished = true")
        println(io, "success = false")
    end
end

function signal_terminate(watch_file)
    open(watch_file, "w") do io
        println(io, "params_updated = true")
        println(io, "terminate = true")
        println(io, "solve_finished = true")
        println(io, "success = false")
    end
end

# # Calibration process
N_iter = 5
@info "Running EK updates for $N_iter iterations"

function perform_ek_update()
    @info "performing EK update..."
    sleep(1)
end

for iteration in 1:N_iter
    @info "Running iteration $iteration"
    @time begin
        @everywhere begin
            myid() ≠ 1 && signal_start_run("WatchFile_$(myid()).jl")
        end
    end
    perform_ek_update()
end

for worker in workers()
    @info "Shutting off worker $worker"
    signal_terminate("WatchFile_$(worker).jl")
end

# cleanup
for worker in workers()
    rm("WatchFile_$(worker).jl"; force=true)
end

