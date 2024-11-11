function Newton(∇ᵏf,init;max_itr=1000,ϵ=1e-4,α=1.0,logging=false,return_solution_path=false)
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
        θ -= α * (∇²f \ ∇f)
        if return_solution_path
            push!(Θ,θ)
        end
    end
    
    return (;solution=θ, status=:reached_max_iteration, solution_path=Θ) 
end

function Exponentialize(B::Vector{Bool},∇ᵏf::Function)
    h(bool::Bool,μ::T) where T<:Real = bool ? exp(μ) : μ
    h⁻¹(bool::Bool,θ::T) where T<:Real  = bool ? log(θ) : θ
    ∂h(bool::Bool,μ::T) where T<:Real = bool ? exp(μ) : 1.0
    ∂²h(bool::Bool,μ::T) where T<:Real = bool ? exp(μ) : 0.0

    h(μ::Vector{T}) where T<:Real = h.(B,μ) 
    ∇h(μ::Vector{T}) where T<:Real = ∂h.(B,μ)
    ∇²h(μ::Vector{T}) where T<:Real = ∂²h.(B,μ)

    return μ -> ( ∇ᵏf(h(μ))  |> ∇ᵏf_val -> (∇ᵏf_val[1] .* ∇h(μ),
                                            diagm(∇²h(μ) .* ∇ᵏf_val[1]) + ∇h(μ)*∇h(μ)' .* ∇ᵏf_val[2])
    )
end