using ArgParse
using CairoMakie
using NCDatasets
using Statistics
const NC = NCDatasets

function quick_plots(ncfile, out_path)
    NC.Dataset(ncfile) do ds
        group_names = keys(ds.group)
        for group_name in group_names
            pName1, pName2 = split(group_name, ".")
            group = ds.group[group_name]  # fetch group data
            xaxis_values = group[pName1][:]  # physical values along x-axis
            yaxis_values = group[pName2][:]  # physical values along y-axis
            nxvals = length(xaxis_values)
            nyvals = length(yaxis_values)

            case_names = group["case"][:]
            n_cases = length(case_names)
            h_res = 600
            fig = Figure(resolution = (h_res * (1 + n_cases), h_res))
            gl_list = [fig[1, i] = GridLayout() for i in 1:n_cases]
            text_size = 23.0
            for (case_ind, case_name) in enumerate(case_names)
                gl = gl_list[case_ind]

                ax1 = Axis(gl[1, 1], title = case_name, titlesize = text_size)
                ax2 = Axis(gl[2, 1])

                z_data = group["loss_data"][:, :, case_ind, :]
                z_mean = nanmean(z_data, dims = 3, drop = true)
                z_std = nanstd(z_data, dims = 3, drop = true)

                if nxvals == 1
                    scatter!(ax1, vec(z_mean), yaxis_values, markersize = 15)
                    scatter!(ax2, vec(z_std), yaxis_values, markersize = 15)
                    linkxaxes!(ax1, ax2)
                elseif nyvals == 1
                    scatter!(ax1, xaxis_values, vec(z_mean), markersize = 15)
                    scatter!(ax2, xaxis_values, vec(z_std), markersize = 15)
                    linkyaxes!(ax1, ax2)
                else
                    hm1 = heatmap!(ax1, xaxis_values, yaxis_values, z_mean)
                    hidexdecorations!(ax1, ticks = false)
                    Colorbar(gl[1:2, 2], hm1)
                    hm2 = heatmap!(ax2, xaxis_values, yaxis_values, z_std, colorrange = extrema(z_mean))
                end

            end  # end cases

            # Sum of all cases
            gl = fig[1, n_cases + 1]

            ax1 = Axis(gl[1, 1], title = "Cases average", titlesize = text_size)
            ax2 = Axis(gl[2, 1])

            z_data = group["loss_data"][:, :, :, :]
            z_case_mean = nanmean(z_data, dims = 3, drop = true)
            z_mean = nanmean(z_case_mean, dims = 3, drop = true)
            z_std = nanstd(z_case_mean, dims = 3, drop = true)

            if nxvals == 1
                scatter!(ax1, vec(z_mean), yaxis_values, markersize = 15)
                scatter!(ax2, vec(z_std), yaxis_values, markersize = 15)
                linkxaxes!(ax1, ax2)
                # axis labels
                xtitle = "Loss ($pName1 = $(only(xaxis_values)))"
                ytitle = pName2
            elseif nyvals == 1
                scatter!(ax1, xaxis_values, vec(z_mean), markersize = 15)
                scatter!(ax2, xaxis_values, vec(z_std), markersize = 15)
                linkyaxes!(ax1, ax2)
                # axis labels
                xtitle = pName1
                ytitle = "Loss ($pName2 = $(only(yaxis_values)))"
            else
                hm1 = heatmap!(ax1, xaxis_values, yaxis_values, z_mean)
                hidexdecorations!(ax1, ticks = false)
                Colorbar(gl[1:2, 2], hm1)
                hm2 = heatmap!(ax2, xaxis_values, yaxis_values, z_std, colorrange = extrema(z_mean))
                # axis labels
                xtitle = pName1
                ytitle = pName2
            end

            Label(fig[2, :], text = string(xtitle), textsize = text_size)  # x-axis
            Label(fig[:, 0], text = string(ytitle), textsize = text_size, rotation = pi / 2)  # y-axis
            save(joinpath(out_path, "$(group_name).png"), fig)
        end  # end groups
    end  # end NC.Dataset
end  # end quick_plots()

# https://discourse.julialang.org/t/nanmean-for-3d-array/62569/4
# -->
nansum((s, n), x) = isnan(x) ? (s, n) : (s + x, n + 1)
division((s, n)) = s / n
nanmean(a; dims = :, drop = false) = begin
    res = division.(reduce(nansum, a, init = (zero(eltype(a)), 0), dims = dims))
    drop ? dropdims(res, dims = dims) : res
end

nanvar(a; dims = :, drop = false) = nanmean((a .- nanmean(a, dims = dims)) .^ 2, dims = dims, drop = drop)
nanstd(a; dims = :, drop = false) = sqrt.(nanvar(a, dims = dims, drop = drop))
# <--


function parse_commandline_lm()

    s = ArgParseSettings(; description = "Run data path input")

    @add_arg_table s begin
        "--sim_dir"
        help = "Grid search simulations directory"
        arg_type = String
    end

    return ArgParse.parse_args(s)  # parse_args(ARGS, s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline_lm()
    sim_dir = args["sim_dir"]
    out_path = joinpath(sim_dir, "figures")
    mkpath(out_path)
    ncfile = joinpath(sim_dir, "loss.nc")
    quick_plots(ncfile, out_path)
end
