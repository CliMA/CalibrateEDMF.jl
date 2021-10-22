@everywhere include("calibrate.jl")

### RUN SIMULATION ###
N_ens = 20 # number of ensemble members
N_iter = 5 # number of EKP iterations.
println("NUMBER OF ENSEMBLE MEMBERS: $N_ens")
println("NUMBER OF ITERATIONS: $N_iter")
run_calibrate(N_ens, N_iter)
