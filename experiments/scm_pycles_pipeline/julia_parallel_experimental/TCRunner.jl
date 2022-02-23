# TCRunner.jl
import TurbulenceConvection

const tc_dir = pkgdir(TurbulenceConvection)
include(joinpath(tc_dir, "driver", "main.jl"))
include(joinpath(tc_dir, "driver", "generate_namelist.jl"))
import .NameList

watch_file = "WatchFile_$(myid()).jl"
@info "Processor $(myid()) monitoring file $watch_file."

function read_watch(watch_file)
    lines = readlines(watch_file)
    params_updated = parse(Bool, strip(last(split(lines[1], "="))))
    terminate      = parse(Bool, strip(last(split(lines[2], "="))))
    solve_finished = parse(Bool, strip(last(split(lines[3], "="))))
    success        = parse(Bool, strip(last(split(lines[4], "="))))
    return (; params_updated, terminate, solve_finished, success)
end

# update watch file after running TC.jl:
function write_watch_post_run(watch_file, return_code)
    success = return_code == :success
    open(watch_file, "w") do io
        println(io, "params_updated = false")
        println(io, "terminate = false")
        println(io, "solve_finished = true")
        println(io, "success = $success")
    end
end

function reset_watch_file(watch_file)
    open(watch_file, "w") do io
        println(io, "params_updated = false")
        println(io, "terminate = false")
        println(io, "solve_finished = false")
        println(io, "success = false")
    end
end
reset_watch_file(watch_file)

# @everywhere function do_work(jobs, results) # define work function everywhere
#    while true
#        job_id = take!(jobs)
#        exec_time = rand()
#        sleep(exec_time) # simulates elapsed time doing actual work
#        put!(results, (job_id, exec_time, myid()))
#    end
# end

# while true
for i in 1:1
    nt = read_watch(watch_file) # continuously watch a file
    if nt.terminate
        # cleanup() # deallocate, close up shop
        # exit() # for real job
        break # for interactive run
    end
    if nt.params_updated
        # namelist has been updated by main processor
        # namelist = read(namelist_file)
        # namelist = NameList.default_namelist("Bomex") # need to replace with read from file
        # ds_tc_filename, return_code = main(namelist)

        sleep(0.1) # don't waste clock cycles
        @info "TC running..."
        sleep(0.1) # don't waste clock cycles
        @info "TC running..."
        sleep(0.1) # don't waste clock cycles


        @info "NC file updated: $ds_tc_filename"
        # NC file now available
        write_watch_post_run(watch_file, return_code)
    end
    # Calibration update occurs in calibrator script on main processor
    sleep(1 #=second=#) # don't waste clock cycles
    @info "Waiting for $watch_file to be updated... Time = $(time())"
end

@info "Processor $(myid()) terminated."
