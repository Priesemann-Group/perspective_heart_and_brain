using Pkg
Pkg.activate(@__DIR__)
using DynamicalSystems
using DifferentialEquations
using CairoMakie
using ChaosTools
using IntervalArithmetic
import IntervalArithmetic.(..)

figurepath = normpath(joinpath(@__DIR__, "../../figures/"))
##
include("funcs.jl")
##
f, w1, w2 = fitzhugh_nagumo_datseris_parlitz(ϵ=0.05, b=0.1, return_nullclines=true)

## get the fixed points
fhn = CoupledODEs(f, [0.0, 0.0], [x-> 0, ])
fp,J = fixedpoints(fhn, IntervalBox(interval(-3,1), interval(-1,1)))
print("fixed points: ", fp[1])

## Draw the state space of the model with nullclines
fig = Figure(size = (300, 300))
ax = Axis(fig[1, 1], xlabel = L"$v$", ylabel = L"$w$", xticks = 0:1:1, yticks = 0:0.4:0.4)
# Draw the nullclines
v = -2:0.01:2
lines!(ax, v, w1.(v), color = :black, linewidth = 2, linestyle = :dot, label = "w1")
lines!(ax, v, w2.(v), color = :orange, linewidth = 2, linestyle = :dot, label = "w2")
xlims!(ax, -0.2, 1.1)
ylims!(ax, -0.2, 0.5)
fig

## single action potential
u0=fp[1]
stimulus(t) = 10<t<12 ? 0.2 : 0.0
p = [stimulus,]
prob = ODEProblem(f, u0, (0,75), p)
sol = solve(prob, dtmax=1.0, saveat=0.01)
t = sol.t;
v = map(x->x[1], sol.u);

##
fig = Figure(size = (400, 300))
ax = Axis(fig[1, 1]; xlabel = "time (a.u)", ylabel = "membrane potential (a.u.)")
lines!(ax, t, v)
# plot non-zero stimulus as gray area in the plot
stimulus_t = t[stimulus.(t) .> 0]
stimulus_v = ones(length(stimulus_t)) * (fp[1][1] - 0.1)
vlines!(ax, stimulus_t, ymin=0.0, ymax=0.05, color = :gray)
# save as pdf
#save(joinpath(figurepath, "action_potential_fitzhugh_nagumo_single.pdf"), fig)
fig

##
stim_intervals = [  (0.0 .. 1.0, 0.0), # no stim in first interval
                    (2.0 .. 3.0, 0.12),
                    (21.0 .. 22.0, 0.2),
                    (51.0 .. 52.0, 0.2)]
stimamp(si) = si[2]
stimstart(si) = si[1].lo
stimend(si) = si[1].hi
responsestart(si) = stimstart(si)
responsestart(sis,i) = responsestart(sis[i])
responseend(sis,i) = i < length(sis) ? sis[i+1][1].lo : ∞

plot_intervals = [responsestart(stim_intervals,i) .. responseend(stim_intervals,i) for i in 1:length(stim_intervals)]
plot_colors = [
        RGBf(0., 0., 0.),
        RGBf(0.7, 0.7, 0),
        RGBf(0.7, 0., 0.7),
        RGBf(0., 0.7, 0.7)
    ]
function I_stim(t)
    for (iv, s) ∈ stim_intervals
        if t ∈ iv
            return s
        end
    end
    return 0.0
end

# fixed point
dtsample = 0.01

u0=fp[1]
p = [I_stim,]
prob = ODEProblem(f, u0, (0,75), p)
sol = solve(prob, dtmax=1.0, saveat=dtsample)
tall,vall,wall = extracttraj(sol)
tp, vp, wp = splittraj(plot_intervals, sol)
##
fig = Figure(size = (400, 300))
axstim = Axis(fig[1, 1]; ylabel = "stimulus", yticks = [0,maximum(stimamp.(stim_intervals))])
ax = Axis(fig[2:5, 1]; xlabel = "time (a.u)", ylabel = "membrane potential (a.u.)")
hidexdecorations!(axstim)
hidespines!(axstim, :r, :b, :t)
ylims!(axstim, maximum(stimamp.(stim_intervals)).*(-0.2, 1.2))
linkxaxes!(ax, axstim)
rowgap!(fig.layout, 1, 8)


# plot membrane potential
for (t,v,w,si,c) in zip(tp, vp, wp, stim_intervals, plot_colors)
    t0 = stimstart(si)
    t1 = stimend(si)
    a = stimamp(si)

    lines!(axstim, [t[1], t0], [0., 0.], color = :black)
    lines!(axstim, [t0, t0, t1, t1], [0., a, a, 0.], color = c)
    lines!(axstim, [t1, t[end]], [0., 0.], color = :black)
    
    if a > 0
        vspan!(ax, si[1].lo, si[1].hi; ymin=0., ymax=0.05, color = :gray)
        vspan!(ax, si[1].lo, si[1].hi; ymin=0.05, ymax=0.95, color = (:gray, 0.3))
        vspan!(ax, si[1].lo, si[1].hi; ymin=0.95, ymax=1.0, color = :gray)
    end
    lines!(ax, t, v, color = c)
end
# save as pdf
save(joinpath(figurepath, "action_potential_fitzhugh_nagumo_multiple.pdf"), fig)
fig

##
# plot phase space
# Draw the state space of the model with nullclines
fig = Figure(size = (400, 300))
ax = Axis(fig[1, 1], xlabel = L"$v$", ylabel = L"$w$")

vstep=0.15
wstep = 0.075
grid_v = -0.4:vstep:1.1 # also used for plot range
grid_w = -0.05:wstep:0.5 # extrema are also used for streamplot
withstream = true # whether to make a streamplot (otherwise, plot arrows on grid)
arrowcolor = RGBAf(0.5, 0.5, 0.5, 0.3)

vrange = extrema(grid_v) .+ (-0.5, 0.5).*vstep
wrange = extrema(grid_w) .+ (-0.5, 0.5).*wstep
xlims!(ax, vrange)
ylims!(ax, wrange)

# Draw the nullclines
vs = -2:0.01:2
lines!(ax, vs, w1.(vs), color = :black, linewidth = 2, linestyle = :dot, label = "w1")
lines!(ax, vs, w2.(vs), color = :orange, linewidth = 2, linestyle = :dot, label = "w2")

if withstream
    streamplot!(ax, (x,y) -> Point2(f([x, y], [x->0,], 0)...), vrange, wrange;
                gridsize = (20,20,20),
                arrow_size = 8, color = x->arrowcolor)
else
    # plot a grid of vectors that encode the strenght of f(v, w)
    for v_ in grid_v, w_ in grid_w
        dv, dw = f([v_, w_], [x->0,], 0)
        # draw errors (dv,dw) at grid points
        arrows!(ax, [v_ - dv*vstep/2], [w_ - dw*wstep/2], [dv*vstep], [dw*wstep];
                    color = arrowcolor, arrowsize=6, align = :origin)
    end
end

for (t,v,w,si,c) in zip(tp, vp, wp, stim_intervals, plot_colors)
    #if we haven't done a stream plot
    #only indicate trajectory direction outside stimuli
    if t ∉ si[1]
        for (v_, w_) in zip(v[1:200:end], w[1:200:end])
            dv, dw = f([v_, w_], [x->0,], 0)
            arrows!(ax, [v_], [w_], [dv], [dw], color = c, align = :head, lengthscale = 1e-5)
        end
    end
    lines!(ax, v, w, color = c)
end

scatter!(ax, [u0[1]], [u0[2]], marker = :circle, color = :red)

save(joinpath(figurepath, "action_potential_fitzhugh_nagumo_multiple_phase"*(withstream ? "_stream" : "")*".pdf"), fig)
fig