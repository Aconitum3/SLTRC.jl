function C(FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    cL,cR = ObservationInterval.left, ObservationInterval.right
    a,_ = params(FX)
    return cR - cL + NumericalIntegration(v -> ccdf(FY,v),Interval(0,cL-a))
end

function logC(FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    return C(FX,FY,ObservationInterval) |> log
end

function ∇C(FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    Yprms = Distributions.params(FY) |> collect
    FY = Fname(FY)
    cL = ObservationInterval.left
    a,_ = params(FX)
    return NumericalIntegration(v -> gradient(θ -> ccdf(FY(θ...),v), Yprms)[1],Interval(0,cL-a))
end

function ∇²C(FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    Yprms = Distributions.params(FY) |> collect
    FY = Fname(FY)
    cL = ObservationInterval.left
    a,_ = params(FX)
    return NumericalIntegration(v -> hessian(θ -> ccdf(FY(θ...),v), Yprms),Interval(0,cL-a))
end

function ∇ᵏylogC(FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T};kwargs...) where {D<:Distribution{Univariate,Continuous},T<:Real}
    C_val = C(FX,FY,ObservationInterval)
    
    ∇logC = ∇C(FX,FY,ObservationInterval) / C_val
    
    ∇²logC = ∇²C(FX,FY,ObservationInterval;kwargs...)  / C_val - ∇logC*∇logC'
    return ∇logC, ∇²logC
end

function logp̃(d::Union{CompleteData,WeaklyLeftTruncatedData},FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    return logpdf(FY,d.failure-d.install)
end

function logp̃(d::Union{RightCensoredData,WeaklyLeftTruncatedRightCensoredData},FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    cR = ObservationInterval.right
    return logccdf(FY,cR-d.install)
end

function logp̃(d::StrictlyLeftTruncatedData,FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    cL = ObservationInterval.left
    a,_ = params(FX)
    return ccdf(FY,d.failure-cL) - ccdf(FY,d.failure-a) |> log
end

function p̃(d::StrictlyLeftTruncatedRightCensoredData,FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T};NumericalIntegration::Function=Default_NumericalIntegration) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    cL,cR = ObservationInterval.left,ObservationInterval.right
    a,_ = params(FX)
    return NumericalIntegration(v -> ccdf(FY,cR-v),Interval(a,cL))
end

function ∇ᵏylogp̃(d::StrictlyLeftTruncatedData,FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D<:Distribution{Univariate,Continuous},T<:Real}
    FYname = Fname(FY)
    Yprms = params(FY) |> collect
    
    ∇logp̃ = gradient(θ -> logp̃(d,FX,FYname(θ...),ObservationInterval), Yprms)[1]
    ∇²logp̃ = hessian(θ -> logp̃(d,FX,FYname(θ...),ObservationInterval), Yprms)
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏylogp̃(d::StrictlyLeftTruncatedRightCensoredData,FX::Uniform,FY::D,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D<:Distribution{Univariate,Continuous},T<:Real}
    cL, cR = ObservationInterval.left, ObservationInterval.right
    a,_ = params(FX)
    FYname = Fname(FY)
    Yprms = params(FY) |> collect
    
    p̃_val = p̃(d,FX,FY,ObservationInterval;NumericalIntegration=NumericalIntegration)
    
    ∇logp̃ = NumericalIntegration(v -> gradient(θ -> ccdf(FYname(θ...),cR-v),Yprms)[1],Interval(a,cL)) / p̃_val
    ∇²logp̃ = NumericalIntegration(v -> hessian(θ -> ccdf(FYname(θ...),cR-v),Yprms),Interval(a,cL)) / p̃_val - ∇logp̃*∇logp̃'
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏloglikelihood(d::LeftTruncatedRightCensoredDataset,FX::Uniform,FY::D;parallel=false,kwargs...) where {D<:Distribution{Univariate,Continuous}}
    return ∇ᵏyloglikelihood(d,FX,FY;parallel=parallel,kwargs...)
end