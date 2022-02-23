# TCRunner.jl
import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

import TurbulenceConvection
import InteractiveUtils

const tc_dir = pkgdir(TurbulenceConvection)
include(joinpath(tc_dir, "driver", "main.jl"))
include(joinpath(tc_dir, "driver", "generate_namelist.jl"))
import .NameList

function TCmain(i::Int)
    tc_runtime = @elapsed begin
        Logging.with_logger(Logging.NullLogger()) do # Silence TC's log
            main1d(NameList.default_namelist("Bomex"); time_run = false)
        end
    end
    println("Proc $(myid()) running job $i finished in $tc_runtime sec.")

    # tc_runtime = 10*rand()
    # sleep(tc_runtime)
end
