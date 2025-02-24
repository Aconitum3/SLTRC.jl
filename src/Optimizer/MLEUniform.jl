function MLE(dataset::LeftTruncatedRightCensoredDataset,FX::Union{ImproperUniform,Uniform},FY::D;logging::Bool=false,kwargs...) where {D<:Distribution{Univariate,Continuous}}
    Bool_positive_constraints = SLTRC.positive_constraint(FY)
    θ_init = Distributions.params(FY) |> collect
    FYname = SLTRC.Fname(FY)
    if logging
        @info "initial values" θ_init
    end
    
    try
        res = Newton(Bool_positive_constraints,θ -> ∇ᵏyloglikelihood(dataset,FX,FYname(θ...);kwargs...),θ_init;logging=logging,kwargs...)
        return (;solution=FYname(res.solution...),status=res.status,solution_path=res.solution_path)
    catch e
        if e isa DomainError     
            return (;solution=FYname(),status=:solution_is_diverged,solution_path=nothing)
        else
            rethrow(e)
        end
    end
end

function MLE(dataset::LeftTruncatedRightCensoredDataset,FX::ImproperUniform,FY::Exponential;kwargs...)
    @assert dataset.ObservationInterval.left == 0 "this program assumes cL is 0.0."
    L = dataset.ObservationInterval.right - dataset.ObservationInterval.left
    TotalLifeTime = 0.0
    n = zeros(Int,4)
    for i in 1:length(dataset.data)
        data = dataset.data[i]
        if isa(data,CompleteData) || isa(data,WeaklyLeftTruncatedData)
            n[1] += 1
            TotalLifeTime += data.failure - data.install
        elseif isa(data,RightCensoredData) || isa(data,WeaklyLeftTruncatedRightCensoredData)
            n[2] += 1
            TotalLifeTime += L - data.install
        elseif isa(data,StrictlyLeftTruncatedData)
            n[3] += 1
            TotalLifeTime += data.failure
        else # isa(data,StrictlyLeftTruncatedRightCensoredData)
            n[4] += 1
            TotalLifeTime += L
        end
    end

    if n[1] == n[2] == n[3] == 0
        return (;solution=nothing,status=:solution_is_not_exist)
    else
        coef1 = - (2n[1] + n[2] + n[3])
        coef2 = TotalLifeTime + (n[4]-n[1]) * L
        coef3 = TotalLifeTime * L
        D = coef2^2 - 4coef1*coef3
        sol1 = (-coef2 + sqrt(D))/2coef1
        sol2 = (-coef2 - sqrt(D))/2coef1
        return (;solution=Exponential(max(sol1,sol2)),status=:optimal_solution)
    end
end