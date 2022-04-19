using Test
using NCDatasets
using CalibrateEDMF.KalmanProcessUtils
import CalibrateEDMF.KalmanProcessUtils: LearningRateScheduler

@testset "KalmanProcessUtils" begin

    @testset "LearningRateScheduler" begin
        Δt_init = 1.0
        τ = 5

        Δt_decay = PiecewiseConstantDecay(Δt_init, τ)
        Δt_growth = PiecewiseConstantGrowth(Δt_init, τ)
        @test isa(Δt_decay, LearningRateScheduler)
        @test isa(Δt_growth, LearningRateScheduler)

        t_future = 10
        @test get_Δt(Δt_decay, t_future) < Δt_init
        @test get_Δt(Δt_growth, t_future) > Δt_init
    end

end
