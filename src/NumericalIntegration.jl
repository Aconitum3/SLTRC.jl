function SimpsonRule(f::Function,I::ClosedInterval{T₁};N::T₂=2^6,kwargs...) where {T₁<:Real, T₂<:Integer} 
    a,b = I.left, I.right
    J = 0*f(a).^0
    if a == b
        return J
    end
    x = a:(b-a)/(2N):b
    y = f.(x)
    for n in 2:N+1
        J += (x[2n-1]-x[2n-3])/6.0 * ( y[2n-3] + 4.0*y[2n-2] + y[2n-1] )
    end
    return J
end

function TrapezoidalRule(f::Function,I::ClosedInterval{T₁};N::T₂=2^6,kwargs...) where {T₁<:Real, T₂<:Integer} 
    a,b = I.left, I.right
    x = a:(b-a)/(N-1):b
    y = f.(x)
    J = 0*f(a).^0
    for n in 2:N
        J += y[n-1] + y[n]
    end

    J = J * (b-a)/(2(N-1))
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

function DETrapezoidalRule(f::Function,I::ClosedInterval{T};kwargs...) where T<:Real
    a,b = I.left,I.right
    if a == b
        return 0*f(a).^0
    end
    return DoublyExponentialize(f,I;kwargs...) |> v -> TrapezoidalRule(v...;kwargs...)
end