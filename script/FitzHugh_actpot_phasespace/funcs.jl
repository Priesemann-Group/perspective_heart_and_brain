# Differential equations of fitzhugh-nagumo model
# References:
# Datseris & Parliz, Nonlinear Dynamics, 2022
# see also: https://link.springer.com/referenceworkentry/10.1007/978-1-4614-7320-6_147-1
# Original: https://www.cell.com/biophysj/pdf/S0006-3495(61)86902-6.pdf
function fitzhugh_nagumo_datseris_parlitz(;a=3, b=0.2, ϵ=0.01, return_nullclines=false)
    function f(u, p, t)
        v, w = u
        I, = p
        dv = a*v*(v-b)*(1-v) - w + I(t)
        dw = ϵ*(v-w)
        return SVector(dv, dw)
    end
    if return_nullclines
        w1(v) = a*v*(v-b)*(1-v)
        w2(v) = v
        return f, w1, w2
    end
    return f
end
function fitzhugh_nagumo_wikipedia(;a=0.7, b=0.8, τ=12.5, return_nullclines=false)
    function f(u, p, t)
        v, w = u
        I, = p
        dv = v - v^3/3 - w + I(t)
        dw = (v + a - b*w)/τ
        return SVector(dv, dw)
    end
    if return_nullclines
        w1(v) = v - v^3/3
        w2(v) = (v + a)/b
        return f, w1, w2
    end
    return f
end

# Extract individual trajectories from solution
function extracttraj(sol)
    numvars = length(sol.u[1])
    return sol.t, [getindex.(sol.u,i) for i in 1:numvars]...
end

# Split trajectory into different parts given by intervals
splittraj(intervals, sol) = splittraj(intervals, extracttraj(sol)...)
function splittraj(intervals, t::AbstractVector{T}, us::Vararg{AbstractVector{T}, N}) where {T, N}
    (length(t) == only(unique(length.(us)))) || error("Lengths of t and the different `u`s are not identical")
    ts = Vector{Vector{T}}()
    uss = [Vector{Vector{T}}() for _ in 1:N]
    for interval in intervals
        idcs = in.(t, interval)
        
        push!(ts, t[idcs])
        for i in 1:N
            push!(uss[i], us[i][idcs])
        end
    end
    return ts, uss...
end
