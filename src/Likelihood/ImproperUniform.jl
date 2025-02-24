
struct ImproperUniform <: ContinuousUnivariateDistribution
end

function C(FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    L = ObservationInterval.right - ObservationInterval.left
    return L + mean(FY) 
end

function logC(FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    return C(FX,FY,ObservationInterval) |> log
end

∇C(FX::ImproperUniform,FY::Exponential,ObservationInterval::ClosedInterval{T}) where {T<:Real} = [1.0]
∇²C(FX::ImproperUniform,FY::Exponential,ObservationInterval::ClosedInterval{T}) where {T<:Real} = [0.0;;]
#==
function ∇C(FX::ImproperUniform,FY::Weibull,ObservationInterval::ClosedInterval{T}) where {T<:Real}
    m,η = Distributions.params(FY)
    g = gamma(1.0+1/m)
    dig = digamma(1.0+1/m)
    return [η*g*dig*(-1/m^2), g]
end
function ∇²C(FX::ImproperUniform,FY::Weibull,ObservationInterval::ClosedInterval{T}) where {T<:Real}
    m,η = Distributions.params(FY)
    g = gamma(1.0+1/m)
    dig = digamma(1.0+1/m)
    trig = trigamma(1.0+1/m)
    ele11 = η * (2/m^3 * g * dig + 1/m^4 * g * (dig + trig))
    ele12 = g*dig*(-1/m^2)
    return [ele11 ele12; 
            ele12 0.0]
end
==#

function ∇C(FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    Yprms = Distributions.params(FY) |> collect
    FY = Fname(FY)
    return gradient(θ -> mean(FY(θ...)), Yprms)[1]
end

function ∇²C(FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    Yprms = Distributions.params(FY) |> collect
    FY = Fname(FY)
    return hessian(θ -> mean(FY(θ...)), Yprms)
end

function ∇ᵏylogC(FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T};kwargs...) where {D<:Distribution{Univariate,Continuous},T<:Real}
    C_val = C(FX,FY,ObservationInterval)
    
    ∇logC = ∇C(FX,FY,ObservationInterval) / C_val
    
    ∇²logC = ∇²C(FX,FY,ObservationInterval;kwargs...)  / C_val - ∇logC*∇logC'
    return ∇logC, ∇²logC
end

function logp̃(d::Union{CompleteData,WeaklyLeftTruncatedData},FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    return logpdf(FY,d.failure-d.install)
end

function logp̃(d::Union{RightCensoredData,WeaklyLeftTruncatedRightCensoredData},FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous},T<:Real}
    cR = ObservationInterval.right
    return logccdf(FY,cR-d.install)
end

function logp̃(d::StrictlyLeftTruncatedData,FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T}) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    return logccdf(FY,d.failure)
end

function logp̃(d::StrictlyLeftTruncatedRightCensoredData,FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T};NumericalIntegration::Function=Default_NumericalIntegration) where {D<:Distribution{Univariate,Continuous}, T<:Real}
    return mean(FY) - NumericalIntegration(v -> ccdf(FY,v),Interval(0.0,L)) |> log
end

function ∇ᵏylogp̃(d::StrictlyLeftTruncatedData,FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D<:Distribution{Univariate,Continuous},T<:Real}
    FYname = Fname(FY)
    Yprms = params(FY) |> collect
    
    ∇logp̃ = gradient(θ -> logccdf(FYname(θ...),d.failure), Yprms)[1]
    ∇²logp̃ = hessian(θ -> logccdf(FYname(θ...),d.failure), Yprms) 
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏylogp̃(d::StrictlyLeftTruncatedRightCensoredData,FX::ImproperUniform,FY::D,ObservationInterval::ClosedInterval{T};NumericalIntegration=Default_NumericalIntegration) where {D<:Distribution{Univariate,Continuous},T<:Real}
    L = ObservationInterval.right - ObservationInterval.left
    FYname = Fname(FY)
    Yprms = params(FY) |> collect
    
    p̃_val = mean(FY) - NumericalIntegration(v -> ccdf(FY,v),Interval(0.0,L))
    ∇logp̃ = (gradient(θ -> mean(FYname(θ...)),Yprms)[1] - NumericalIntegration(v -> gradient(θ -> ccdf(FYname(θ...),v),Yprms)[1],Interval(0.0,L))) / p̃_val
    ∇²logp̃ = (hessian(θ -> mean(FYname(θ...)),Yprms)    - NumericalIntegration(v -> hessian(θ -> ccdf(FYname(θ...),v),Yprms),Interval(0.0,L))) / p̃_val - ∇logp̃*∇logp̃'
    return ∇logp̃, ∇²logp̃
end

function ∇ᵏloglikelihood(d::LeftTruncatedRightCensoredDataset,FX::ImproperUniform,FY::D;parallel=false,kwargs...) where {D<:Distribution{Univariate,Continuous}}
    return ∇ᵏyloglikelihood(d,FX,FY;parallel=parallel,kwargs...)
end