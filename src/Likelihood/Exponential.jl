function C(FX::Exponential{T₁},FY::Exponential{T₁},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real,T₂<:Real}
    cL = ObservationInterval.left
    cR = ObservationInterval.right
    θX, θY = params(FX)[1], params(FY)[1]
    
    if cR == Inf
        return exp(-cL/θX) + θY/(θY-θX) * (exp(-cL/θY) - exp(-cL/θX))
    else
        return exp(-cL/θX) - exp(-cR/θX) + θY/(θY-θX) * (exp(-cL/θY) - exp(-cL/θX))
    end
end

function logC(FX::Exponential{T₁},FY::Exponential{T₁},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real,T₂<:Real}
    cL = ObservationInterval.left
    cR = ObservationInterval.right
    
    if cL == 0.0
        return logcdf(FX,cR)
    else
        return log(C(FX,FY,ObservationInterval))
    end
end

function ∇C(FX::Exponential{T₁},FY::Exponential{T₁},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real,T₂<:Real}
    θX, θY = params(FX)[1], params(FY)[1]

    return gradient(θ -> C(Exponential(θ[1]),Exponential(θ[2]),ObservationInterval),[θX,θY])[1]
end

function ∇xC(FX::Exponential{T₁},FY::Exponential{T₁},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real,T₂<:Real}
    θX = params(FX)[1]

    return gradient(θ -> C(Exponential(θ),FY,ObservationInterval),θX)[1]
end

function ∇yC(FX::Exponential{T₁},FY::Exponential{T₁},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real,T₂<:Real}
    θY = params(FY)[1]
    return gradient(θ -> C(FX,Exponential(θ),ObservationInterval),θY)[1]
end

function ∇²C(FX::Exponential{T₁},FY::Exponential{T₁},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real,T₂<:Real}
    θX, θY = params(FX)[1], params(FY)[1]

    return hessian(θ -> C(Exponential(θ[1]),Exponential(θ[2]),ObservationInterval),[θX,θY])
end

function ∇²xC(FX::Exponential{T₁},FY::Exponential{T₁},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real,T₂<:Real}
    θX = params(FX)[1]

    return hessian(θ -> C(Exponential(θ),FY,ObservationInterval),θX)
end

function ∇²yC(FX::Exponential{T₁},FY::Exponential{T₁},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real,T₂<:Real}
    θY = params(FY)[1]
    return hessian(θ -> C(FX,Exponential(θ),ObservationInterval),θY)
end

function logp̃(d::StrictlyLeftTruncatedData,FX::Exponential{T₁},FY::Exponential{T₂},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real,T₂<:Real}
    cL = ObservationInterval.left
    θX,θY = params(FX)[1],params(FY)[1]
    return - d.failure/θY + log( (1-exp(-(θY-θX)/(θX*θY) * cL))/(θY-θX) )
end

function logp̃(d::StrictlyLeftTruncatedRightCensoredData,FX::Exponential{T₁},FY::Exponential{T₂},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real,T₂<:Real}
    cL = ObservationInterval.left
    cR = ObservationInterval.right
    θX,θY = params(FX)[1],params(FY)[1]
    return log(θY) - cR/θY + log( (1-exp(-(θY-θX)/(θX*θY) * cL))/(θY-θX) )
end