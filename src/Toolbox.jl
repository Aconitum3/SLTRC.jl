function ratio(d::LeftTruncatedRightCensoredDataset)
    n = length(d.data)
    output = Dict(
                  :Complete => sum(map(v -> isa(v,CompleteData),d.data))/n,
                  :RightCensored => sum(map(v -> isa(v,RightCensoredData),d.data))/n,
                  :LeftTruncated => sum(map(v -> isa(v,StrictlyLeftTruncatedData),d.data))/n,
                  :LeftTruncatedRightCensored => sum(map(v -> isa(v,StrictlyLeftTruncatedRightCensoredData),d.data))/n
                )
    return output
end

function ratio(FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T};NumericalIntegration=DESimpsonRule,include_unobservable=true) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL, cR = ObservationInterval.left, ObservationInterval.right
    output = Dict(:Unobservable => 0.0,:Complete => 0.0, :RightCensored => 0.0, :LeftTruncated => 0.0, :LeftTruncatedRightCensored => 0.0)
    C_val = C(FX,FY,ObservationInterval;NumericalIntegration=NumericalIntegration)
    output[:Unobservable] = 1.0 - C_val
    if cR == Inf
        output[:Complete] = ccdf(FX,cL)
        output[:LeftTruncated] = NumericalIntegration(v -> pdf(FX,v)*ccdf(FY,cL-v),Interval(0,cL))
    else
        output[:Complete] = NumericalIntegration(v -> pdf(FX,v)*cdf(FY,cR-v),ObservationInterval)
        output[:RightCensored] = NumericalIntegration(v -> pdf(FX,v)*ccdf(FY,cR-v),ObservationInterval)
        output[:LeftTruncated] = NumericalIntegration(v -> pdf(FX,v)*( cdf(FY,cR-v) - cdf(FY,cL-v) ),Interval(0,cL))
        output[:LeftTruncatedRightCensored] = cdf(FX,cL) - NumericalIntegration(v -> pdf(FX,v)*cdf(FY,cR-v),Interval(0,cL))
    end
    if include_unobservable
        return output
    else
        output[:Complete] /= C_val
        output[:RightCensored] /= C_val
        output[:LeftTruncated] /= C_val
        output[:LeftTruncatedRightCensored] /= C_val
        filter!(v -> v[1] != :Unobservable, output)
        return output
    end
end

function positive_constraint(F::D) where D<:Distribution{Univariate,Continuous} 
    res = Dict(
        :Exponential => [true],
        :Gamma => [true,true],
        :LogNormal => [false,true],
        :Uniform => [false,false],
        :Weibull => [true,true],
    )
    return res[Symbol(Fname(F))]
end