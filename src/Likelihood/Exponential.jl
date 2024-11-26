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

function E∇²loglikelihood(FX::Exponential{T₁},FY::Exponential{T₁},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real, T₂<: Real}
    μx = params(FX)[1]
    μy = params(FY)[1]
    cL, cR = ObservationInterval.left ,ObservationInterval.right
    
    C_val = C(FX,FY,ObservationInterval)
    
    eLx, eLy = exp(-cL/μx), exp(-cL/μy)
    eRx, eRy = exp(-cR/μx), exp(-cR/μy)
    eRLy = exp(-(cR-cL)/μy)

    y_yx = μy/(μy-μx)
    x_yx = μx/(μy-μx)

    # All of the following terms must be divided by C_val
    #===================================================#
    
    # P( Data is Complete )
    PC = (1 - y_yx * eRLy)*eLx + x_yx * eRx
    # E[Y1(Data is Complete)]
    EY_C = μy*(eLx-eRx) - y_yx * (cR + μy - μx*y_yx) * (eLx*eRLy-eRx) + y_yx * (cL*eLx*eRLy-cR*eRx)
    
    # P( Data is Left-Truncated )
    PLT = y_yx * (eLy-eLx-eRy+eLx*eRLy)
    # [Z1(Data is Left-Truncated)] 
    EZ_LT = y_yx * (cL*eLy-cR*eRy + μy*(eLy-eRy)) * (1-eLx/eLy)
    
    # P( Data is Right-Censored )
    PRC = y_yx * (eLx*eRLy-eRx)
    # E[X1(Data is Right-Censored)]
    EX_RC = y_yx * (cL*eLx*eRLy - cR*eRx + μx*y_yx * (eLx*eRLy-eRx)) 
    
    # P( Data is Left-Truncated and Right-Censored )
    PLTRC = y_yx * eRy * (1-eLx/eLy)
    
    #===================================================#

    ∇²A = hessian(v->log(1/(v-μx)*(1-exp(-cL*(1/μx-1/v)))),μy)
    ∇²B = ∇ᵏylogC(FX,FY,ObservationInterval)[2]
    
    C_term  = -2/μy^3 * EY_C + (1/μy^2-∇²B)*PC
    LT_term = - 2/μy^3 * EZ_LT + (∇²A-∇²B)*PLT
    RC_term = 2/μy^3 * EX_RC + (-2cR/μy^3-∇²B)*PRC
    LTRC_term = (∇²A - 2cR/μy^3 - 1/μy^2 - ∇²B)*PLTRC
    
    return (C_term+LT_term+RC_term+LTRC_term)/C_val
end

function E∇²conditionalloglikelihood(FX::Exponential{T₁},FY::Exponential{T₁},ObservationInterval::ClosedInterval{T₂}) where {T₁<:Real, T₂<:Real}
    μx, μy = params(FX)[1], params(FY)[1]
    cL, cR = ObservationInterval.left, ObservationInterval.right

    eLx = exp(-cL/μx)
    eRx = exp(-cR/μx)
    eRLy = exp(-(cR-cL)/μy)

    first_term = 1/μy * 1/(μy-μx) * (eLx*eRLy - eRx)
    second_term = -1/μy^2 * (1-eRx - (1-eLx)*eRLy)
    return (first_term+second_term)/(1-eRx)
end
