function MLE(dataset::LeftTruncatedRightCensoredDataset,FX::D₁,FY::D₂;logging::Bool=false,kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous}}
    Bool_positive_constraints = [positive_constraint(FX);positive_constraint(FY)]
    θ_init = [params(FX)...;params(FY)...]
    FXname, FYname = Fname(FX), Fname(FY)
    len_Xprms, len_Yprms = length(params(FX)), length(params(FY))
    
    res = Newton(Bool_positive_constraints,θ -> ∇ᵏloglikelihood(dataset,FXname(θ[1:len_Xprms]...),FYname(θ[end-len_Yprms+1:end]...);kwargs...),θ_init;logging=logging,kwargs...)
    return (;solution=(FXname(res.solution[1:len_Xprms]...),FYname(res.solution[end-len_Yprms+1:end]...)),status=res.status,solution_path=res.solution_path)
end

function MLE_Alternative(dataset::LeftTruncatedRightCensoredDataset,FX::D₁,FY::D₂;ϵ=1e-4,alt_max_itr=100,logging::Bool=false,kwargs...) where {D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous}}
    Bool_X_positive_constraints = positive_constraint(FX)
    Bool_Y_positive_constraints = positive_constraint(FY)
    θX_init, θY_init = params(FX) |> collect, params(FY) |> collect
    FXname, FYname = Fname(FX), Fname(FY)
    len_Xprms, len_Yprms = length(params(FX)), length(params(FY))


    index_complete = findall(v -> isa(v,CompleteData),dataset.data)
    index_rightcensored = findall(v -> isa(v,RightCensoredData),dataset.data) 
    
    # estimate initial θX,θY
    RightCensoredDataset = LeftTruncatedRightCensoredDataset(dataset.data[[index_complete;index_rightcensored]],dataset.ObservationInterval)
    res = Newton(Bool_X_positive_constraints,θ -> ∇ᵏxloglikelihood(RightCensoredDataset,FXname(θ...),FY;kwargs...),θX_init;kwargs...)
    θX = res.solution
    if res.status != :converged_local_maximal
        return (;solution=(FXname(θX...),FY), status=res.status)
    end
    
    res = Newton(Bool_Y_positive_constraints,θ -> ∇ᵏyloglikelihood(RightCensoredDataset,FXname(θX...),FYname(θ...);kwargs...),θY_init;kwargs...)
    θY = res.solution
    
    if res.status != :converged_local_maximal
        return (;solution=(FXname(θX...),FYname(θY...)), status=res.status)
    end
    
    if logging
        @info "initial values" [FXname(θX...),FYname(θY...)]
    end

    θX_path = zeros(alt_max_itr+1,len_Xprms)
    θY_path = zeros(alt_max_itr+1,len_Yprms)
    θX_path[1,:] = θX
    θY_path[1,:] = θY
    
    # estimate θX, θY alternatively
    for i in 2:alt_max_itr+1
        
        resX = Newton(Bool_X_positive_constraints,θ -> ∇ᵏxloglikelihood(dataset,FXname(θ...),FYname(θY...);kwargs...),θX;kwargs...)
        θX = resX.solution
        resY = Newton(Bool_X_positive_constraints,θ -> ∇ᵏxloglikelihood(dataset,FXname(θX...),FYname(θ...);kwargs...),θY;kwargs...)
        θY = resY.solution

        if norm(θX_path[i-1,:] - θX) < ϵ && norm(θY_path[i-1,:] - θY) < ϵ
            hess = ∇ᵏloglikelihood(dataset,FXname(θX...),FYname(θY...))[2]
            if in(false, eigvals(hess) .< -ϵ)
                return (;solution=(FXname(θX...),FYname(θY...)),status=:converged_suddle_point,solution_path=[θX_path[1:i] θY_path[1:i]])
            else
                return (;solution=(FXname(θX...),FYname(θY...)),status=:converged_local_maximal,solution_path=[θX_path[1:i] θY_path[1:i]])
            end
        end
        θX_path[i,:] = θX
        θY_path[i,:] = θY

        if logging
            @info "$i :" [FXname(θX...),FYname(θY...)]
        end

    end
    return (;solution=(FXname(θX...),FYname(θY...)),status=:reached_max_iteration,solution_path=[θX_path θY_path])
end

function ConditionalMLE(dataset::LeftTruncatedRightCensoredDataset,FY::D;logging::Bool=false,kwargs...) where D<:Distribution{Univariate,Continuous}
    Bool_positive_constraints = positive_constraint(FY)
    θ_init = params(FY) |> collect
    FYname = Fname(FY)
    
    index_NOT_StrictlyLeftTruncated = findall(v -> (!isa)(v,StrictlyLeftTruncatedData) && (!isa)(v,StrictlyLeftTruncatedRightCensoredData), dataset.data)
    if length(index_NOT_StrictlyLeftTruncated) != length(dataset.data) 
        @warn "This Dataset inclues Strictly Left-Truncated Data. Strictly Left-Truncated Data are excluded automatically."
        dataset = LeftTruncatedRightCensoredDataset(dataset.data[index_NOT_StrictlyLeftTruncated], dataset.ObservationInterval)
    end
      
    res = Newton(Bool_positive_constraints,θ -> ∇ᵏconditionalloglikelihood(dataset,FYname(θ...);kwargs...),θ_init;logging=logging,kwargs...)
    return (;solution=(FYname(res.solution...)),status=res.status,solution_path=res.solution_path)
end
