function RightCensoredExponentialMLE(dataset::LeftTruncatedRightCensoredDataset) where {I<:Integer, T<:Real}
    cR = dataset.ObservationInterval.right
    index_complete = findall(v -> isa(v,CompleteData),dataset.data)
    index_rightcensored = findall(v -> isa(v,RightCensoredData),dataset.data) 
    
    install_complete = dataset.data[index_complete] .|> v -> v.install
    install_rightcensored = dataset.data[index_rightcensored] .|> v -> v.install
    lifetime_complete = dataset.data[index_complete] .|> v -> v.failure - v.install
    lifetime_rightcensored = dataset.data[index_rightcensored] .|> v -> cR - v.install
    
    μx = ( sum(install_complete) + sum(install_rightcensored) ) / (length(lifetime_complete) + length(lifetime_rightcensored)) 
    μy = ( sum(lifetime_complete) + sum(lifetime_rightcensored) ) / length(lifetime_complete)
    return (;solution=[Exponential(μx),Exponential(μy)],status=:converged_local_maximal)
end
