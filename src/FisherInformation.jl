function FisherInformation(FX::Uniform,FY::Exponential,ObservationInterval::ClosedInterval)
    a,b = Distributions.params(FX)
    μ = Distributions.params(FY)[1]
    cL,cR = ObservationInterval.left, ObservationInterval.right

    L = cR - cL
    K = cL - a


    Cons = 1 / ( L + μ*(1-exp(-K/μ)) )

    EY_Complete = Cons * ( L*μ*exp(-L/μ) - 2μ^2 + 2μ^2*exp(-L/μ) + L*μ )
    
    EX_RC = Cons * ( cR*μ - cL*μ*exp(-L/μ) - μ^2 + μ^2*exp(-L/μ) )
    
    EX_LT = Cons * μ * ( (cL-μ)*exp(cL/μ) - (a-μ)*exp(a/μ) ) * ( exp(-cL/μ)-exp(-cR/μ) )
    EZ_LT = Cons * (exp(cL/μ) - exp(a/μ)) * (-cR*μ*exp(-cR/μ) + cL*μ*exp(-cL/μ) - μ^2*(exp(-cR/μ) - exp(-cL/μ)))
    EY_LT = EZ_LT - EX_LT

    EX_LTRC = Cons * μ * ( (cL-μ)*exp(cL/μ) - (a-μ)*exp(a/μ) ) * exp(-cR/μ) 
    
    P = ratio(FX,FY,ObservationInterval;include_unobservable=false)
    
    A(u) = log( L + u*(1-exp(-K/μ)) )
    B(u) = log( exp(cL/u) - exp(a/u) )
    
    A2 = hessian(A,μ)[1,1]
    B2 = hessian(B,μ)[1,1]

    return Dict(
        :X_Censored => 2/μ^3 * EY_Complete - 1/μ^2 * P[:Complete] + 2/μ^3 * (cR*P[:RightCensored]-EX_RC) + 2/μ^3 * EZ_LT - B2*P[:LeftTruncated] + P[:LeftTruncatedRightCensored] * (-B2 + 2cR/μ^3 + 1/μ^2) + A2,
        :X_Uncensored => 2/μ^3 * (EY_Complete+EY_LT) - 1/μ^2 * (P[:Complete]+P[:LeftTruncated]) - 2/μ^3 * (EX_RC+EX_LTRC) + 2cR/μ^3 * (P[:RightCensored]+P[:LeftTruncatedRightCensored]) + A2
    )
end

function FisherInformation(FX::ImproperUniform,FY::Exponential,ObservationInterval::ClosedInterval)
    μ = Distributions.params(FY)[1]
    cL,cR = ObservationInterval.left, ObservationInterval.right

    L = cR - cL
    
    Cons = 1/μ^2 * 1/(L+μ)
    return Dict(
        :X_Censored => Cons * (L+μ-μ^2/(L+μ)),
        :X_Uncensored => Cons * (L+2μ-μ^2/(L+μ))
    )
end

function FisherInformation_Allison(FX::Uniform,FY::Exponential,ObservationInterval::ClosedInterval)
    a,b = Distributions.params(FX)
    μ = Distributions.params(FY)[1]
    cL,cR = ObservationInterval.left, ObservationInterval.right

    L = cR - cL
    K = cL - a


    Cons = 1 / ( L + μ*(1-exp(-K/μ)) )

    EY_Complete = Cons * ( L*μ*exp(-L/μ) - 2μ^2 + 2μ^2*exp(-L/μ) + L*μ )
    
    EX_RC = Cons * ( cR*μ - cL*μ*exp(-L/μ) - μ^2 + μ^2*exp(-L/μ) )
    
    P = ratio(FX,FY,ObservationInterval;include_unobservable=false)
    
    return 2/μ^3 * EY_Complete - 1/μ^2 * P[:Complete] + 2/μ^3 * (cR*P[:RightCensored]-EX_RC) 
end

function FisherInformation_Allison(FX::ImproperUniform,FY::Exponential,ObservationInterval::ClosedInterval)
    μ = Distributions.params(FY)[1]
    cL,cR = ObservationInterval.left, ObservationInterval.right

    L = cR - cL
    
    return 1/μ^2 * 1/(L+μ) * (L-μ+μ*exp(-L/μ))
end

function FisherInformation_CLM(FX::Uniform,FY::Exponential,ObservationInterval::ClosedInterval)
    a,b = Distributions.params(FX)
    μ = Distributions.params(FY)[1]
    cL,cR = ObservationInterval.left, ObservationInterval.right

    L = cR - cL
    K = cL - a

    Cons = 1 / ( L + μ*(1-exp(-K/μ)) )

    return Cons * 1/μ^2 * ( L - μ * (1-exp(-L/μ)) * exp(-K/μ) )
end

function FisherInformation_CLM(FX::ImproperUniform,FY::Exponential,ObservationInterval::ClosedInterval)
    μ = Distributions.params(FY)[1]
    cL,cR = ObservationInterval.left, ObservationInterval.right

    L = cR - cL


    return L/(L+μ) * 1/μ^2
end

function FisherInformationMC(K,L,FY::D;N=1000,
    dataset = sampling(N,Uniform(-K,L),FY,Interval(0,L)),
    NumericalIntegration=Default_NumericalIntegration,
    kwargs...) where D<:Distribution{Univariate,Continuous}

    FYname = Fname(FY)
    Yprms = params(FY) |> collect
    m = length(Yprms)

    I_Base = zeros(m,m)
    I_C = zeros(m,m)
    I_UC = zeros(m,m)
    I_C_approx = zeros(m,m)
    I_CLM = zeros(m,m)

    n_LTRC = 0
    #Threads.@threads 
    for i in 1:N
        d = dataset.WeaklyLeftTruncatedRightCensoredDataset.data[i]
        if isa(d, CompleteData)
            lifetime = d.failure - d.install
            if lifetime < 1e-100
                lifetime = 1e-100
            end
            val = hessian(θ -> logpdf(FYname(θ...),lifetime),Yprms)
            I_Base += val
        elseif isa(d, RightCensoredData)
            val = hessian(θ -> logccdf(FYname(θ...),L-d.install),Yprms)
            I_Base += val
        elseif isa(d, WeaklyLeftTruncatedData)
            lifetime = d.failure - d.install
            if lifetime < 1e-100
                lifetime = 1e-100
            end
            val = hessian(θ -> logpdf(FYname(θ...),lifetime),Yprms)
            I_C += hessian(θ -> log(ccdf(FYname(θ...),d.failure) - ccdf(FYname(θ...),d.failure + K)),Yprms)
            I_C_approx += hessian(θ -> logccdf(FYname(θ...),d.failure),Yprms)
            I_UC += val
            I_CLM += val - hessian(θ -> logccdf(FYname(θ...),- d.install),Yprms)
        else # isa(d, WeaklyLeftTruncatedRightCensoredData)
            val = hessian(θ -> logccdf(FYname(θ...),L-d.install),Yprms)
            I_UC += val
            I_CLM += val - hessian(θ -> logccdf(FYname(θ...),- d.install),Yprms)
            n_LTRC += 1
        end
    end
    val_ltrc = begin
        if n_LTRC == 0
            zeros(m,m)
        else
            p̃_val = NumericalIntegration(v -> ccdf(FY,v),Interval(L,K+L))
            ∇logp̃ = (NumericalIntegration(v -> gradient(θ -> ccdf(FYname(θ...),v),Yprms)[1],Interval(L,K+L))) / p̃_val
            
            NumericalIntegration(v -> hessian(θ -> ccdf(FYname(θ...),v),Yprms),Interval(L,K+L)) / p̃_val - ∇logp̃*∇logp̃'
        end
    end
    val_ltrc_approx = begin
        p̃_val = mean(FY) - NumericalIntegration(v -> ccdf(FY,v),Interval(0.0,L))
        ∇logp̃ = (gradient(θ -> mean(FYname(θ...)),Yprms)[1] - NumericalIntegration(v -> gradient(θ -> ccdf(FYname(θ...),v),Yprms)[1],Interval(0.0,L))) / p̃_val
        
        (hessian(θ -> mean(FYname(θ...)),Yprms) - NumericalIntegration(v -> hessian(θ -> ccdf(FYname(θ...),v),Yprms),Interval(0.0,L))) / p̃_val - ∇logp̃*∇logp̃'
    end
    val_cons = begin
        h_val = L + NumericalIntegration(v -> ccdf(FY,v),Interval(0.0,K))
        ∇LogH = NumericalIntegration(v -> gradient(θ -> ccdf(FYname(θ...),v),Yprms)[1],Interval(0.0,K)) / h_val
        
        NumericalIntegration(v -> hessian(θ -> ccdf(FYname(θ...),v),Yprms),Interval(0.0,K)) / h_val - ∇LogH*∇LogH'
    end
    val_cons_approx = ∇ᵏlogC(ImproperUniform(),FY,Interval(0,L))[2]

    return Dict(
        :X_Censored => -1/N * (I_Base + I_C + n_LTRC * val_ltrc) + val_cons,
        :X_Censored_Approx => -1/N * (I_Base + I_C_approx + n_LTRC * val_ltrc_approx) + val_cons_approx,
        :X_Uncensored => -1/N * (I_Base + I_UC) + val_cons,
        :X_Uncensored_Approx => -1/N * (I_Base + I_UC) + val_cons_approx,
        :CLM => -1/N * (I_Base + I_CLM),
        :Allison => -1/N * I_Base
    )
end