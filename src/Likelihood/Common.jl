function C(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration::Function=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    cR = ObservationInterval.right
    return cdf(FX,cR) - NumericalIntegration(v -> pdf(FX,v)*cdf(FY,cL-v),Interval(0,cL))
end

function logC(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    cR = ObservationInterval.right
    
    if cL == 0.0
        return logcdf(FX,cR)
    else
        return log(C(FX,FY,ObservationInterval;kwargs...))
    end
end

function ∇C(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL, cR = ObservationInterval.left, ObservationInterval.right
    FXname, FYname = Fname(FX), Fname(FY)
    Xprms, Yprms = params(FX), params(FY)
    len_Xprms, len_Yprms =  length(Xprms), length(Yprms)
    prms = [Xprms...;Yprms...]

    if cR == Inf
        return - NumericalIntegration( v -> gradient(θ -> pdf(FXname(θ[1:len_Xprms]...),v)*cdf(FYname(θ[end-len_Yprms+1:end]...),cL-v),prms)[1], Interval(0.0,cL))
    else
        return gradient(θ -> cdf(FXname(θ[1:len_Xprms]...),cR),prms)[1] - NumericalIntegration( v -> gradient(θ -> pdf(FXname(θ[1:len_Xprms]...),v)*cdf(FYname(θ[end-len_Yprms+1:end]...),cL-v),prms)[1], Interval(0.0,cL))
    end
end

function ∇xC(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL, cR = ObservationInterval.left, ObservationInterval.right
    
    FXname = Fname(FX)
    Xprms = params(FX) |> collect
    if cR == Inf
        return - NumericalIntegration( v -> gradient(θ -> pdf(FXname(θ...),v)*cdf(FY,cL-v),Xprms)[1], Interval(0.0,cL))
    else
        return gradient(θ -> cdf(FXname(θ...),cR),Xprms)[1] - NumericalIntegration( v -> gradient(θ -> pdf(FXname(θ...),v)*cdf(FY,cL-v),Xprms)[1], Interval(0.0,cL))
    end
end

function ∇yC(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    FYname = Fname(FY)
    Yprms = params(FY) |> collect

    return - NumericalIntegration( v -> gradient(θ -> pdf(FX,v)*cdf(FYname(θ...),cL-v),Yprms)[1], Interval(0.0,cL))
end

function ∇²C(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL, cR = ObservationInterval.left, ObservationInterval.right
    FXname, FYname = Fname(FX), Fname(FY)
    Xprms, Yprms = params(FX), params(FY)
    len_Xprms, len_Yprms =  length(Xprms), length(Yprms)
    prms = [Xprms...;Yprms...]

    if cR == Inf
        return - NumericalIntegration( v -> hessian(θ -> pdf(FXname(θ[1:len_Xprms]...),v)*cdf(FYname(θ[end-len_Yprms+1:end]...),cL-v),prms), Interval(0.0,cL))
    else
        return hessian(θ -> cdf(FXname(θ[1:len_Xprms]...),cR),prms) - NumericalIntegration( v -> hessian(θ -> pdf(FXname(θ[1:len_Xprms]...),v)*cdf(FYname(θ[end-len_Yprms+1:end]...),cL-v),prms), Interval(0.0,cL))
    end
end

function ∇²xC(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL, cR = ObservationInterval.left, ObservationInterval.right
    FXname = Fname(FX)
    Xprms = params(FX) |> collect
    if cR == Inf
        return - NumericalIntegration( v -> hessian(θ -> pdf(FXname(θ...),v)*cdf(FY,cL-v),Xprms), Interval(0.0,cL))
    else
        return hessian(θ -> cdf(FXname(θ...),cR),Xprms) - NumericalIntegration( v -> hessian(θ -> pdf(FXname(θ...),v)*cdf(FY,cL-v),Xprms), Interval(0.0,cL))
    end
end

function ∇²yC(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    FYname = Fname(FY)
    Yprms = params(FY) |> collect

    return - NumericalIntegration( v -> hessian(θ -> pdf(FX,v)*cdf(FYname(θ...),cL-v),Yprms), Interval(0.0,cL))
end



function ∇ᵏlogC(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    C_val = C(FX,FY,ObservationInterval;kwargs...)
    
    ∇logC = ∇C(FX,FY,ObservationInterval;kwargs...) / C_val
    
    ∇²logC = ∇²C(FX,FY,ObservationInterval;kwargs...)  / C_val - ∇logC*∇logC'
    return ∇logC, ∇²logC
end

function ∇ᵏxlogC(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    C_val = C(FX,FY,ObservationInterval;kwargs...)
    
    ∇logC = ∇xC(FX,FY,ObservationInterval;kwargs...) / C_val
    
    ∇²logC = ∇²xC(FX,FY,ObservationInterval;kwargs...)  / C_val - ∇logC*∇logC'
    return ∇logC, ∇²logC
end

function ∇ᵏylogC(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    C_val = C(FX,FY,ObservationInterval;kwargs...)
    
    ∇logC = ∇yC(FX,FY,ObservationInterval;kwargs...) / C_val
    
    ∇²logC = ∇²yC(FX,FY,ObservationInterval;kwargs...)  / C_val - ∇logC*∇logC'
    return ∇logC, ∇²logC
end

function ∇logC(FX::D₁ ,FY::D₂,ObservationInterval::ClosedInterval{T};kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    C_val = C(FX,FY,ObservationInterval;kwargs...)
    
    ∇logC = ∇C(FX,FY,ObservationInterval;kwargs...) / C_val
    
    return ∇logC
end

function ∇xlogC(FX::D₁ ,FY::D₂,ObservationInterval::ClosedInterval{T};kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    C_val = C(FX,FY,ObservationInterval;kwargs...)
    
    ∇logC = ∇xC(FX,FY,ObservationInterval;kwargs...) / C_val
    
    return ∇logC
end

function ∇ylogC(FX::D₁ ,FY::D₂,ObservationInterval::ClosedInterval{T};kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    C_val = C(FX,FY,ObservationInterval;kwargs...)
    
    ∇logC = ∇yC(FX,FY,ObservationInterval;kwargs...) / C_val
    
    return ∇logC
end

function p̃(d::Union{CompleteData,WeaklyLeftTruncatedData},FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T}) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    return pdf(FX,d.install) * pdf(FY,d.failure-d.install)
end

function p̃x(d::Union{CompleteData,WeaklyLeftTruncatedData},FX::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    return pdf(FX,d.install)
end

function p̃y(d::Union{CompleteData,WeaklyLeftTruncatedData},FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    return pdf(FY,d.failure - d.install)
end

function p̃(d::Union{RightCensoredData,WeaklyLeftTruncatedRightCensoredData},FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T}) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cR = ObservationInterval.right
    return pdf(FX,d.install) * ccdf(FY,cR-d.install)
end

function p̃x(d::Union{RightCensoredData,WeaklyLeftTruncatedRightCensoredData},FX::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    return pdf(FX,d.install)
end

function p̃y(d::Union{RightCensoredData,WeaklyLeftTruncatedRightCensoredData},FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    cR = ObservationInterval.right
    return ccdf(FY,cR-d.install)
end

function p̃(d::StrictlyLeftTruncatedData,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration::Function=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    return NumericalIntegration(v -> pdf(FX,v)*pdf(FY,d.failure-v),Interval(0.0,cL))
end

function p̃(d::StrictlyLeftTruncatedRightCensoredData,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration::Function=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    cR = ObservationInterval.right
    return cdf(FX,cL) - NumericalIntegration(v -> pdf(FX,v)*cdf(FY,cR-v),Interval(0.0,cL))
end

function logp̃(d::Union{CompleteData,WeaklyLeftTruncatedData},FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T}) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    return logpdf(FX,d.install) + logpdf(FY,d.failure-d.install)
end

function logp̃x(d::Union{CompleteData,WeaklyLeftTruncatedData},FX::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    return logpdf(FX,d.install)
end

function logp̃y(d::Union{CompleteData,WeaklyLeftTruncatedData},FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    return logpdf(FY,d.failure-d.install)
end

function logp̃(d::Union{RightCensoredData,WeaklyLeftTruncatedRightCensoredData},FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T}) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cR = ObservationInterval.right
    return logpdf(FX,d.install) + logccdf(FY,cR-d.install)
end

function logp̃x(d::Union{RightCensoredData,WeaklyLeftTruncatedRightCensoredData},FX::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    return logpdf(FX,d.install)
end

function logp̃y(d::Union{RightCensoredData,WeaklyLeftTruncatedRightCensoredData},FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    cR = ObservationInterval.right
    return logccdf(FY,cR-d.install)
end

function logp̃(d::StrictlyLeftTruncatedData,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration::Function=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    return NumericalIntegration(v -> pdf(FX,v)*pdf(FY,d.failure-v),Interval(0.0,cL)) |> log
end

function logp̃(d::StrictlyLeftTruncatedRightCensoredData,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration::Function=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    cR = ObservationInterval.right
    return cdf(FX,cL) - NumericalIntegration(v -> pdf(FX,v)*cdf(FY,cR-v),Interval(0.0,cL)) |> log
end

function ∇ᵏlogp̃(d::Union{CompleteData,RightCensoredData,WeaklyLeftTruncatedData,WeaklyLeftTruncatedRightCensoredData},FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    FXname, FYname = Fname(FX), Fname(FY)
    Xprms, Yprms = params(FX), params(FY)
    len_Xprms, len_Yprms =  length(Xprms), length(Yprms)
    prms = [Xprms...;Yprms...]
    
    ∇logp̃ = gradient(θ -> logp̃(d,FXname(θ[1:len_Xprms]...),FYname(θ[end-len_Yprms+1:end]...),ObservationInterval), prms)[1]
    ∇²logp̃ = hessian(θ -> logp̃(d,FXname(θ[1:len_Xprms]...),FYname(θ[end-len_Yprms+1:end]...),ObservationInterval), prms)
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏlogp̃x(d::Union{CompleteData,RightCensoredData,WeaklyLeftTruncatedData,WeaklyLeftTruncatedRightCensoredData},FX::D,ObservationInterval::ClosedInterval{T};kwargs...) where {D<:Distribution{Univariate,Continuous},T<:Real}
    FXname = Fname(FX)
    Xprms = params(FX) |> collect
    
    ∇logp̃ = gradient(θ -> logp̃x(d,FXname(θ...),ObservationInterval), Xprms)[1]
    ∇²logp̃ = hessian(θ -> logp̃x(d,FXname(θ...),ObservationInterval), Xprms)
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏlogp̃y(d::Union{CompleteData,RightCensoredData,WeaklyLeftTruncatedData,WeaklyLeftTruncatedRightCensoredData},FY::D,ObservationInterval::ClosedInterval{T};kwargs...) where {D<:Distribution{Univariate,Continuous},T<:Real}
    FYname = Fname(FY)
    Yprms = params(FY) |> collect
    
    ∇logp̃ = gradient(θ -> logp̃y(d,FYname(θ...),ObservationInterval), Yprms)[1]
    ∇²logp̃ = hessian(θ -> logp̃y(d,FYname(θ...),ObservationInterval), Yprms)
    return ∇logp̃, ∇²logp̃
end

∇ᵏxlogp̃(d::Union{CompleteData,RightCensoredData,WeaklyLeftTruncatedData,WeaklyLeftTruncatedRightCensoredData},
    FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real} = ∇ᵏlogp̃x(d,FX,ObservationInterval;kwargs...)

∇ᵏylogp̃(d::Union{CompleteData,RightCensoredData,WeaklyLeftTruncatedData,WeaklyLeftTruncatedRightCensoredData},
    FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real} = ∇ᵏlogp̃y(d,FY,ObservationInterval;kwargs...)

function ∇ᵏlogp̃(d::StrictlyLeftTruncatedData,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    FXname, FYname = Fname(FX), Fname(FY)
    Xprms, Yprms = params(FX), params(FY)
    len_Xprms, len_Yprms =  length(Xprms), length(Yprms)
    prms = [Xprms...;Yprms...]
    
    p̃_val = p̃(d,FX,FY,ObservationInterval;NumericalIntegration=NumericalIntegration)
    
    ∇logp̃ = NumericalIntegration(v -> gradient(θ -> pdf(FXname(θ[1:len_Xprms]...),v)*pdf(FYname(θ[end-len_Yprms+1:end]...),d.failure-v), prms)[1],Interval(0.0,cL)) / p̃_val
    ∇²logp̃ = NumericalIntegration(v -> hessian(θ -> pdf(FXname(θ[1:len_Xprms]...),v)*pdf(FYname(θ[end-len_Yprms+1:end]...),d.failure-v), prms),Interval(0.0,cL)) / p̃_val - ∇logp̃*∇logp̃'
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏxlogp̃(d::StrictlyLeftTruncatedData,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    FXname = Fname(FX)
    Xprms = params(FX) |> collect
    
    p̃_val = p̃(d,FX,FY,ObservationInterval;NumericalIntegration=NumericalIntegration)
    
    ∇logp̃ = NumericalIntegration(v -> gradient(θ -> pdf(FXname(θ...),v)*pdf(FY,d.failure-v), Xprms)[1],Interval(0.0,cL)) / p̃_val
    ∇²logp̃ = NumericalIntegration(v -> hessian(θ -> pdf(FXname(θ...),v)*pdf(FY,d.failure-v), Xprms),Interval(0.0,cL)) / p̃_val - ∇logp̃*∇logp̃'
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏylogp̃(d::StrictlyLeftTruncatedData,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    FYname = Fname(FY)
    Yprms = params(FY) |> collect
    
    p̃_val = p̃(d,FX,FY,ObservationInterval;NumericalIntegration=NumericalIntegration)
    
    ∇logp̃ = NumericalIntegration(v -> gradient(θ -> pdf(FX,v)*pdf(FYname(θ...),d.failure-v), Yprms)[1],Interval(0.0,cL)) / p̃_val
    ∇²logp̃ = NumericalIntegration(v -> hessian(θ -> pdf(FX,v)*pdf(FYname(θ...),d.failure-v), Yprms),Interval(0.0,cL)) / p̃_val - ∇logp̃*∇logp̃'
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏlogp̃(d::StrictlyLeftTruncatedRightCensoredData,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL, cR = ObservationInterval.left, ObservationInterval.right
    FXname, FYname = Fname(FX), Fname(FY)
    Xprms, Yprms = params(FX), params(FY)
    len_Xprms, len_Yprms =  length(Xprms), length(Yprms)
    prms = [Xprms...;Yprms...]
    
    p̃_val = p̃(d,FX,FY,ObservationInterval;NumericalIntegration=NumericalIntegration)
    
    ∇logp̃ = ( gradient(θ -> cdf(FXname(θ[1:len_Xprms]...),cL),prms)[1] - NumericalIntegration(v -> gradient(θ -> pdf(FXname(θ[1:len_Xprms]...),v)*cdf(FYname(θ[end-len_Yprms+1:end]...),cR-v),prms)[1],Interval(0.0,cL)) ) / p̃_val
    ∇²logp̃ =( hessian(θ -> cdf(FXname(θ[1:len_Xprms]...),cL),prms) - NumericalIntegration(v -> hessian(θ -> pdf(FXname(θ[1:len_Xprms]...),v)*cdf(FYname(θ[end-len_Yprms+1:end]...),cR-v),prms),Interval(0.0,cL)) ) / p̃_val - ∇logp̃*∇logp̃'
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏxlogp̃(d::StrictlyLeftTruncatedRightCensoredData,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL, cR = ObservationInterval.left, ObservationInterval.right
    FXname = Fname(FX)
    Xprms = params(FX) |> collect
    
    p̃_val = p̃(d,FX,FY,ObservationInterval;NumericalIntegration=NumericalIntegration)
    
    ∇logp̃ = (gradient(θ -> cdf(FXname(θ...),cL),Xprms)[1] - NumericalIntegration(v -> gradient(θ -> pdf(FXname(θ...),v)*cdf(FY,cR-v),Xprms)[1],Interval(0.0,cL)) ) / p̃_val
    ∇²logp̃ = (hessian(θ -> cdf(FXname(θ...),cL),Xprms) - NumericalIntegration(v -> hessian(θ -> pdf(FXname(θ...),v)*cdf(FY,cR-v),Xprms),Interval(0.0,cL)) ) / p̃_val - ∇logp̃*∇logp̃'
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏylogp̃(d::StrictlyLeftTruncatedRightCensoredData,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL, cR = ObservationInterval.left, ObservationInterval.right
    FYname = Fname(FY)
    Yprms = params(FY) |> collect
    
    p̃_val = p̃(d,FX,FY,ObservationInterval;NumericalIntegration=NumericalIntegration)
    
    ∇logp̃ = - NumericalIntegration(v -> gradient(θ -> pdf(FX,v)*cdf(FYname(θ...),cR-v),Yprms)[1],Interval(0.0,cL)) / p̃_val
    ∇²logp̃ = - NumericalIntegration(v -> hessian(θ -> pdf(FX,v)*cdf(FYname(θ...),cR-v),Yprms),Interval(0.0,cL)) / p̃_val - ∇logp̃*∇logp̃'
    return ∇logp̃, ∇²logp̃
end

function loglikelihood(d::LeftTruncatedRightCensoredDataset,FX::D₁,FY::D₂;kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous}}
    ObservationInterval = d.ObservationInterval
    data = d.data
    
    indexes_NOT_StrictlyLeftTruncatedRightCensored = findall(v -> (!isa)(v,StrictlyLeftTruncatedRightCensoredData),data)
    n_StrictlyLeftTruncatedRightCensored = length(data) - length(indexes_NOT_StrictlyLeftTruncatedRightCensored)

    ∑logp̃ = 0.0
    
    if n_StrictlyLeftTruncatedRightCensored != 0
        ∑logp̃ += n_StrictlyLeftTruncatedRightCensored * logp̃(StrictlyLeftTruncatedRightCensoredData(),FX,FY,ObservationInterval;kwargs...)
    end
    
    for i in indexes_NOT_StrictlyLeftTruncatedRightCensored
        ∑logp̃ += logp̃(data[i],FX,FY,ObservationInterval;kwargs...)
    end

    loglikelihood = ∑logp̃ - length(data) * logC(FX,FY,ObservationInterval;kwargs...)
    return loglikelihood
end

function ∇ᵏloglikelihood(d::LeftTruncatedRightCensoredDataset,FX::D₁,FY::D₂;kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous}}
    ObservationInterval = d.ObservationInterval
    data = d.data
    
    indexes_NOT_StrictlyLeftTruncatedRightCensored = findall(v -> (!isa)(v,StrictlyLeftTruncatedRightCensoredData),data)
    n_StrictlyLeftTruncatedRightCensored = length(data) - length(indexes_NOT_StrictlyLeftTruncatedRightCensored)

    len_prms = length(params(FX)) + length(params(FY))
    ∑∇ᵏlogp̃ = (zeros(len_prms), zeros(len_prms,len_prms))
    
    if n_StrictlyLeftTruncatedRightCensored != 0
        ∑∇ᵏlogp̃ = ∑∇ᵏlogp̃ .+ n_StrictlyLeftTruncatedRightCensored .* ∇ᵏlogp̃(StrictlyLeftTruncatedRightCensoredData(),FX,FY,ObservationInterval;kwargs...)
    end
    
    for i in indexes_NOT_StrictlyLeftTruncatedRightCensored
        ∑∇ᵏlogp̃ = ∑∇ᵏlogp̃ .+ ∇ᵏlogp̃(data[i],FX,FY,ObservationInterval;kwargs...)
    end

    ∇ᵏloglikelihood = ∑∇ᵏlogp̃ .- length(data) .* ∇ᵏlogC(FX,FY,ObservationInterval;kwargs...)
    return ∇ᵏloglikelihood
end

function ∇ᵏxloglikelihood(d::LeftTruncatedRightCensoredDataset,FX::D₁,FY::D₂;kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous}}
    ObservationInterval = d.ObservationInterval
    data = d.data
    
    indexes_NOT_StrictlyLeftTruncatedRightCensored = findall(v -> (!isa)(v,StrictlyLeftTruncatedRightCensoredData),data)
    n_StrictlyLeftTruncatedRightCensored = length(data) - length(indexes_NOT_StrictlyLeftTruncatedRightCensored)

    len_prms = length(params(FX))
    ∑∇ᵏlogp̃ = (zeros(len_prms), zeros(len_prms,len_prms))
    
    if n_StrictlyLeftTruncatedRightCensored != 0
        ∑∇ᵏlogp̃ = ∑∇ᵏlogp̃ .+ n_StrictlyLeftTruncatedRightCensored .* ∇ᵏxlogp̃(StrictlyLeftTruncatedRightCensoredData(),FX,FY,ObservationInterval;kwargs...)
    end

    for i in indexes_NOT_StrictlyLeftTruncatedRightCensored
        ∑∇ᵏlogp̃ = ∑∇ᵏlogp̃ .+ ∇ᵏxlogp̃(data[i],FX,FY,ObservationInterval;kwargs...)
    end

    if len_prms > 1
        ∇ᵏloglikelihood = ∑∇ᵏlogp̃ .- length(data) .* ∇ᵏxlogC(FX,FY,ObservationInterval;kwargs...)
        return ∇ᵏloglikelihood
    else
        ∇ᵏloglikelihood = (∑∇ᵏlogp̃[1][1], ∑∇ᵏlogp̃[2][1,1]) .- length(data) .* ∇ᵏxlogC(FX,FY,ObservationInterval;kwargs...)
        return ∇ᵏloglikelihood
    end
end

function ∇ᵏyloglikelihood(d::LeftTruncatedRightCensoredDataset,FX::D₁,FY::D₂;parallel=false,kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous}}
    ObservationInterval = d.ObservationInterval
    data = d.data
    
    indexes_NOT_StrictlyLeftTruncatedRightCensored = findall(v -> (!isa)(v,StrictlyLeftTruncatedRightCensoredData),data)
    n_StrictlyLeftTruncatedRightCensored = length(data) - length(indexes_NOT_StrictlyLeftTruncatedRightCensored)

    len_prms = length(params(FY))
    ∑∇ᵏlogp̃ = (zeros(len_prms), zeros(len_prms,len_prms))
    
    if n_StrictlyLeftTruncatedRightCensored != 0
        ∑∇ᵏlogp̃ = ∑∇ᵏlogp̃ .+ n_StrictlyLeftTruncatedRightCensored .* ∇ᵏylogp̃(StrictlyLeftTruncatedRightCensoredData(),FX,FY,ObservationInterval;kwargs...)
    end
    if parallel
        len_NOT_SLTRC = length(indexes_NOT_StrictlyLeftTruncatedRightCensored)
        ∇logp̃ = zeros(len_prms,len_NOT_SLTRC)
        ∇²logp̃ = zeros(len_prms,len_prms,len_NOT_SLTRC) 
        
        Threads.@threads for i in 1:len_NOT_SLTRC
            ∇logp̃[:,i], ∇²logp̃[:,:,i] = ∇ᵏylogp̃(data[indexes_NOT_StrictlyLeftTruncatedRightCensored[i]],FX,FY,ObservationInterval)
        end

        ∑∇ᵏlogp̃ = ∑∇ᵏlogp̃ .+ ( dropdims(sum(∇logp̃,dims=2),dims=2),dropdims(sum(∇²logp̃,dims=3),dims=3) )
    else
        for i in indexes_NOT_StrictlyLeftTruncatedRightCensored
            ∑∇ᵏlogp̃ = ∑∇ᵏlogp̃ .+ ∇ᵏylogp̃(data[i],FX,FY,ObservationInterval;kwargs...)
        end
    end
    
    if len_prms > 1
        ∇ᵏloglikelihood = ∑∇ᵏlogp̃ .- length(data) .* ∇ᵏylogC(FX,FY,ObservationInterval;kwargs...)
        return ∇ᵏloglikelihood
    else
        ∇ᵏloglikelihood = (∑∇ᵏlogp̃[1][1], ∑∇ᵏlogp̃[2][1,1]) .- length(data) .* ∇ᵏylogC(FX,FY,ObservationInterval;kwargs...)
        return ∇ᵏloglikelihood
    end
end

function conditionalloglikelihood(d::LeftTruncatedRightCensoredDataset,FY::D;kwargs...) where D<:Distribution{Univariate,Continuous}
    ObservationInterval = d.ObservationInterval
    cL = ObservationInterval.left
    data = d.data
    
    conditionalloglikelihood = 0.0
    
    for i in 1:length(data)
        conditionalloglikelihood += logp̃y(data[i],FY,ObservationInterval;kwargs...)
        if isa(data[i], WeaklyLeftTruncatedData) || isa(data[i], WeaklyLeftTruncatedRightCensoredData)
            conditionalloglikelihood -= logccdf(FY,cL - data[i].install)
        end
    end

    return conditionalloglikelihood
end

function ∇ᵏconditionalloglikelihood(d::LeftTruncatedRightCensoredDataset,FY::D;parallel=false,kwargs...) where D<:Distribution{Univariate,Continuous}
    ObservationInterval = d.ObservationInterval
    cL = ObservationInterval.left
    data = d.data


    FYname = Fname(FY)
    Yprms = params(FY) |> collect
    len_prms = length(Yprms)
    ∇ᵏconditionalloglikelihood = (zeros(len_prms), zeros(len_prms,len_prms))
    
    if parallel
        ∇conditionallogli = zeros(len_prms,length(data))
        ∇²conditionallogli = zeros(len_prms,len_prms,length(data))
        Threads.@threads for i in 1:length(data)
            ∇conditionallogli[:,i], ∇²conditionallogli[:,:,i] = ∇ᵏlogp̃y(data[i],FY,ObservationInterval;kwargs...)
            if isa(data[i], WeaklyLeftTruncatedData) || isa(data[i], WeaklyLeftTruncatedRightCensoredData)
                τ = cL - data[i].install
                ∇conditionallogli[:,i] -= gradient(θ -> logccdf(FYname(θ...), τ), Yprms)[1]
                ∇²conditionallogli[:,:,i] -= hessian(θ -> logccdf(FYname(θ...), τ), Yprms)
            end
        end

        ∇ᵏconditionalloglikelihood = ( dropdims(sum(∇conditionallogli,dims=2),dims=2), dropdims(sum(∇²conditionallogli,dims=3),dims=3) )
    else          
        for i in 1:length(data)
            ∇ᵏconditionalloglikelihood = ∇ᵏconditionalloglikelihood .+ ∇ᵏlogp̃y(data[i],FY,ObservationInterval;kwargs...)
            if isa(data[i], WeaklyLeftTruncatedData) || isa(data[i], WeaklyLeftTruncatedRightCensoredData)
                τ = cL - data[i].install
                ∇ᵏconditionalloglikelihood = ∇ᵏconditionalloglikelihood .- ( gradient(θ -> logccdf(FYname(θ...), τ), Yprms)[1], hessian(θ -> logccdf(FYname(θ...), τ), Yprms) )
            end
        end
    end

    return ∇ᵏconditionalloglikelihood
end