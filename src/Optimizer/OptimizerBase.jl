function Newton(∇ᵏf::Function,init::Vector{T₁};max_itr::I=1000,ϵ::T₂=1e-4,α::T₃=1.0,logging::Bool=false,return_solution_path::Bool=false,kwargs...) where {I<:Integer,T₁<:Real,T₂<:Real,T₃<:Real}
    θ = init
    Θ = Vector{<:Real}[θ]
    for _ in 1:max_itr
        ∇f, ∇²f = ∇ᵏf(θ)
        ∇f_norm = norm(∇f)
        if logging
            @info "∇f: $(∇f .|> v -> round(v,digits=2)), |∇f|: $(round(∇f_norm,digits=5)) det∇²f: $(round(det(∇²f),digits=4))"
        end
        if ∇f_norm < ϵ
            if logging
                @info "converged"
                @info "eigvals(∇²f): $(eigvals(∇²f) .|> v -> round(v,digits=4))"
            end
            if (eigvals(∇²f) .< -ϵ) == ones(length(θ))
                return (;solution=θ, status=:converged_local_maximal, solution_path=Θ)
            else
                return (;solution=θ, status=:converged_suddle_point, solution_path=Θ)
            end
        end
        if abs(det(∇²f)) < ϵ
            return (;solution=θ, status=:det∇²f_is_zero, solution_path=Θ)
        end
        if in(true, isnan.(∇f))
            return (;solution=θ, status=:∇f_is_diverged, solution_path=Θ)
        end
        if in(true, isnan.(∇²f))
            return (;solution=θ, status=:∇²f_is_diverged, solution_path=Θ)
        end
        θ -= α * (∇²f \ ∇f)
        if return_solution_path
            push!(Θ,θ)
        end
    end
    
    return (;solution=θ, status=:reached_max_iteration, solution_path=Θ) 
end


function Exponentialize(B::Vector{Bool},∇ᵏf::Function)
    h(bool::Bool,μ::T) where T<:Real = bool ? exp(μ) : μ
    ∂h(bool::Bool,μ::T) where T<:Real = bool ? exp(μ) : 1.0
    ∂²h(bool::Bool,μ::T) where T<:Real = bool ? exp(μ) : 0.0

    h(μ::Vector{T}) where T<:Real = h.(B,μ) 
    ∇h(μ::Vector{T}) where T<:Real = ∂h.(B,μ)
    ∇²h(μ::Vector{T}) where T<:Real = ∂²h.(B,μ)

    return μ -> ( ∇ᵏf(h(μ))  |> ∇ᵏf_val -> (∇ᵏf_val[1] .* ∇h(μ),
                                            diagm(∇²h(μ) .* ∇ᵏf_val[1]) + ∇h(μ)*∇h(μ)' .* ∇ᵏf_val[2])
    )
end

function Newton(B::Vector{Bool},∇ᵏf::Function,init::Vector{T};kwargs...) where T<:Real
    
    h(bool::Bool,μ::T) where T<:Real = bool ? exp(μ) : μ
    h⁻¹(bool::Bool,θ::T) where T<:Real  = bool ? log(θ) : θ
    
    ∇ᵏf_exponentialize = Exponentialize(B,∇ᵏf)
    res = Newton(∇ᵏf_exponentialize,h⁻¹.(B,init);kwargs...)
    return (;solution=h.(B,res.solution), status=res.status, solution_path=map(θ -> h.(B,θ),res.solution_path))
end

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