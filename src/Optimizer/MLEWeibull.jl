function RightCensoredWeibullMLE(dataset::LeftTruncatedRightCensoredDataset;max_itr::I=10000,ϵ::T=1e-4) where {I<:Integer, T<:Real}
    cR = dataset.ObservationInterval.right
    index_complete = findall(v -> isa(v,CompleteData),dataset.data)
    index_rightcensored = findall(v -> isa(v,RightCensoredData),dataset.data) 
    
    install_complete = dataset.data[index_complete] .|> v -> v.install
    install_rightcensored = dataset.data[index_rightcensored] .|> v -> v.install
    lifetime_complete = dataset.data[index_complete] .|> v -> v.failure - v.install
    lifetime_rightcensored = dataset.data[index_rightcensored] .|> v -> cR - v.install
    
    function est(t,δ)
        d = sum(δ)
        x̄ = sum(log.(t).*δ)/d
        m,η = zeros(2)
        m₀ = 1/(log(maximum(t))-x̄)
        h(m) = sum(t.^m.*log.(t))/sum(t.^m) - 1/m - x̄
        
        function ∇h(m)
            a = t.^m / sum(t.^m)
            b = log.(t)
            return sum(a.*b.^2) - sum(a.*b)^2 + 1/m^2
        end
        
        for _ in 1:max_itr
            m = m₀ - h(m₀)/∇h(m₀)
            di = abs((m-m₀)/m)
            if  di < ϵ
                η = (sum(t.^m)/d)^(1/m)
                return (;solution=Weibull(m,η),status=:converged_local_maximal)
            else
            end
            m₀ = m
        end
        return (;solution=Weibull(m,η),status=:reached_max_iteration)
    end

    t = [install_complete;install_rightcensored]
    δ = ones(length(index_complete)+length(index_rightcensored))
    resX = est(t,δ)
    t = [lifetime_complete;lifetime_rightcensored]
    δ = [ones(length(index_complete));zeros(length(index_rightcensored))]
    resY = est(t,δ)
    return (;solution=[resX.solution,resY.solution])
end

function MLE(dataset::LeftTruncatedRightCensoredDataset,FX::Weibull{T},FY::Weibull{T};logging::Bool=false,kwargs...) where T<:Real
    
    θ_init = [params(FX)...;params(FY)...]
    if FX == Weibull() && FY == Weibull()
        # estimate initial θX,θY
        
        index_complete = findall(v -> isa(v,CompleteData),dataset.data)
        index_rightcensored = findall(v -> isa(v,RightCensoredData),dataset.data) 

        RightCensoredDataset = LeftTruncatedRightCensoredDataset(dataset.data[[index_complete;index_rightcensored]],dataset.ObservationInterval)
        
        res = RightCensoredWeibullMLE(RightCensoredDataset)
        θX = params(res.solution[1]) |> collect
        θY = params(res.solution[2]) |> collect
        θ_init = [θX;θY]
    end
    if logging
        @info "initial values" θ_init
    end
    res = Newton([true,true,true,true],θ -> ∇ᵏloglikelihood(dataset,Weibull(θ[1:2]...),Weibull(θ[3:4]...);kwargs...),θ_init;logging=logging,kwargs...)
    return (;solution=(Weibull(res.solution[1:2]...),Weibull(res.solution[3:4]...)),status=res.status,solution_path=res.solution_path)
end

function MLE_Alternative(dataset::LeftTruncatedRightCensoredDataset,FX::Weibull{T},FY::Weibull{T};ϵ=1e-4,alt_max_itr=100,logging::Bool=false,kwargs...) where T<:Real
    θX_init, θX_init = params(FX) |> collect, params(FY) |> collect

    index_complete = findall(v -> isa(v,CompleteData),dataset.data)
    index_rightcensored = findall(v -> isa(v,RightCensoredData),dataset.data) 
    
    # estimate initial θX,θY
    θX = θX_init
    RightCensoredDataset = LeftTruncatedRightCensoredDataset(dataset.data[[index_complete;index_rightcensored]],dataset.ObservationInterval)
    
    res = RightCensoredWeibullMLE(RightCensoredDataset)
    θX = params(res.solution[1]) |> collect
    θY = params(res.solution[2]) |> collect
    
    if logging
        @info "initial values" res.solution 
    end

    θX_path = zeros(alt_max_itr+1,2)
    θY_path = zeros(alt_max_itr+1,2)
    θX_path[1,:] = θX
    θY_path[1,:] = θY
    
    # estimate θX, θY alternatively
    for i in 2:alt_max_itr+1
        
        resX = Newton([true,true],θ -> ∇ᵏxloglikelihood(dataset,Weibull(θ...),Weibull(θY...);kwargs...),θX;kwargs...)
        θX = resX.solution
        resY = Newton([true,true],θ -> ∇ᵏyloglikelihood(dataset,Weibull(θX...),Weibull(θ...);kwargs...),θY;kwargs...)
        θY = resY.solution
        
        if norm(θX_path[i-1,:] - θX) < ϵ && norm(θY_path[i-1,:] - θY) < ϵ
            hess = ∇ᵏloglikelihood(dataset,Weibull(θX...),Weibull(θY...))[2]
            #@info "Hessian" hess
            if in(false, eigvals(hess) .< -ϵ)
                return (;solution=(Weibull(θX...),Weibull(θY...)),status=:converged_suddle_point,solution_path=[θX_path[1:i] θY_path[1:i]])
            else
                return (;solution=(Weibull(θX...),Weibull(θY...)),status=:converged_local_maximal,solution_path=[θX_path[1:i] θY_path[1:i]])
            end
        end
        θX_path[i,:] = θX
        θY_path[i,:] = θY

        if logging
            @info "$i :" [Weibull(θX...),Weibull(θY...)]
        end
    end
    return (;solution=(Weibull(θX...),Weibull(θY...)),status=:reached_max_iteration,solution_path=[θX_path θY_path])
end

function ConditionalMLE(dataset::LeftTruncatedRightCensoredDataset,FY::Weibull;logging::Bool=false,kwargs...)
    θ_init = params(FY) |> collect
    if FY == Weibull()
        # estimate initial θX,θY
        
        index_complete = findall(v -> isa(v,CompleteData),dataset.data)
        index_rightcensored = findall(v -> isa(v,RightCensoredData),dataset.data) 

        RightCensoredDataset = LeftTruncatedRightCensoredDataset(dataset.data[[index_complete;index_rightcensored]],dataset.ObservationInterval)
        
        res = RightCensoredWeibullMLE(RightCensoredDataset)
        θ_init = params(res.solution[2]) |> collect
    end
    
    if logging
        @info "initial values" θ_init 
    end
    
    
    index_NOT_StrictlyLeftTruncated = findall(v -> (!isa)(v,StrictlyLeftTruncatedData) && (!isa)(v,StrictlyLeftTruncatedRightCensoredData), dataset.data)
    if length(index_NOT_StrictlyLeftTruncated) != length(dataset.data) 
        @warn "This Dataset inclues Strictly Left-Truncated Data. Strictly Left-Truncated Data are excluded automatically."
        dataset = LeftTruncatedRightCensoredDataset(dataset.data[index_NOT_StrictlyLeftTruncated], dataset.ObservationInterval)
    end
      
    res = Newton([true,true],θ -> ∇ᵏconditionalloglikelihood(dataset,Weibull(θ...);kwargs...),θ_init;logging=logging,kwargs...)
    return (;solution=(Weibull(res.solution...)),status=res.status,solution_path=res.solution_path)
end