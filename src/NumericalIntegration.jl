function SimpsonRule(f::Function,I::ClosedInterval{T₁};N::T₂=2^6,kwargs...) where {T₁<:Real, T₂<:Integer} 
    a,b = I.left, I.right
    J = 0*f(a).^0
    if a == b
        return J
    end
    x = a:(b-a)/(N-1):b
    for n in 2:N
        J += (x[n]-x[n-1])/6.0 * ( f(x[n-1]) + 4.0*f((x[n-1]+x[n])/2.0) + f(x[n]))
    end
    return J
end

function Standardize(f::Function,I::ClosedInterval{T}) where T<:Real
    a,b = I.left, I.right
    return z -> f(a+(b-a)*(1+z)/2)*(b-a)/2, -1, 1
end

function DoublyExponentialize(f::Function,I::ClosedInterval{T};bound::Real=3.0,kwargs...) where T<:Real
    f_standardized = Standardize(f,I)[1]
    ϕ(t) = tanh(π/2 * sinh(t))
    dϕ(t) = π/2 * cosh(t) * sech(π/2 * sinh(t))^2
    return z -> f_standardized(ϕ(z))*dϕ(z), ClosedInterval(-bound,bound)
end

function DESimpsonRule(f::Function,I::ClosedInterval{T};kwargs...) where T<:Real
    a,b = I.left,I.right
    if a == b
        return 0*f(a).^0
    end
    return DoublyExponentialize(f,I;kwargs...) |> v -> SimpsonRule(v...;kwargs...)
end