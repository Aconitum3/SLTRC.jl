function sampling(n::I,FX::D₁,FY::D₂,ObservationInterval::ClosedInterval{T}) where {I<:Integer,D₁<:Distribution{Univariate,Continuous},D₂<:Distribution{Univariate,Continuous},T<:Real}
    cL = ObservationInterval.left
    cR = ObservationInterval.right

    output_weakly = Vector{LeftTruncatedRightCensoredData}()
    output_strictly = Vector{LeftTruncatedRightCensoredData}()
    while length(output_weakly) < n
        X, Y= rand(FX), rand(FY)
        
        if Interval(X,X+Y) ⊆ Interval(-Inf,cL) || Interval(X,X+Y) ⊆ Interval(cR,Inf)
            continue
        else
            if Interval(X,X+Y) ⊆ ObservationInterval
                push!(output_weakly,CompleteData(X,X+Y))
                push!(output_strictly,CompleteData(X,X+Y))
            elseif X ∈ ObservationInterval && X+Y ∈ Interval(cR,Inf)
                push!(output_weakly,RightCensoredData(X))
                push!(output_strictly,RightCensoredData(X))
            elseif X+Y ∈ ObservationInterval && X ∈ Interval(-Inf,cL)
                push!(output_weakly,WeaklyLeftTruncatedData(X,X+Y))
                push!(output_strictly,StrictlyLeftTruncatedData(X+Y))
            else # ObservationInterval ⊆ Interval(X,X+Y)
                push!(output_weakly,WeaklyLeftTruncatedRightCensoredData(X))
                push!(output_strictly,StrictlyLeftTruncatedRightCensoredData())
            end
        end
    end
    
    return (;
        WeaklyLeftTruncatedRightCensoredDataset=LeftTruncatedRightCensoredDataset(output_weakly,ObservationInterval),
        StrictlyLeftTruncatedRightCensoredDataset=LeftTruncatedRightCensoredDataset(output_strictly,ObservationInterval)
    )
end