
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
    return mean(FY) - NumericalIntegration(v -> ccdf(FY,v),Interval(0,L)) |> log
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
    
    p̃_val = mean(FY) - NumericalIntegration(v -> ccdf(FY,v))
    ∇logp̃ = (gradient(θ -> mean(FYname(θ...)))[1] - NumericalIntegration(v -> gradient(θ -> cdf(FYname(θ...),v),Yprms)[1],Interval(0,L))) / p̃_val
    ∇²logp̃ = (hessian(θ -> mean(FYname(θ...))) - NumericalIntegration(v -> hessian(θ -> cdf(FYname(θ...),cR-v),Yprms),Interval(0.0,L))) / p̃_val - ∇logp̃*∇logp̃'
    return ∇logp̃, ∇²logp̃
end

#==
function ∇ᵏloglikelihood(d::LeftTruncatedRightCensoredDataset,FX::ImproperUniform,FY::D;parallel=false,kwargs...) where {D<:Distribution{Univariate,Continuous}}
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
        ∇ᵏloglikelihood = (∑∇ᵏlogp̃[1], ∑∇ᵏlogp̃[2]) .- length(data) .* ∇ᵏylogC(FX,FY,ObservationInterval;kwargs...)
        return ∇ᵏloglikelihood
    end
end
==#
function ∇ᵏloglikelihood(d::LeftTruncatedRightCensoredDataset,FX::ImproperUniform,FY::D;parallel=false,kwargs...) where {D<:Distribution{Univariate,Continuous}}
    return ∇ᵏyloglikelihood(d,FX,FY;parallel=parallel,kwargs...)
end