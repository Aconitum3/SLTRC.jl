abstract type LeftTruncatedRightCensoredData end

struct CompleteData{T<:Real}<:LeftTruncatedRightCensoredData
    install::T
    failure::T
end

struct RightCensoredData{T<:Real}<:LeftTruncatedRightCensoredData
    install::T
end

struct StrictlyLeftTruncatedData{T<:Real}<:LeftTruncatedRightCensoredData
    failure::T
end

struct WeaklyLeftTruncatedData{T<:Real}<:LeftTruncatedRightCensoredData
    install::T
    failure::T
end

struct StrictlyLeftTruncatedRightCensoredData<:LeftTruncatedRightCensoredData
end

struct WeaklyLeftTruncatedRightCensoredData{T<:Real}<:LeftTruncatedRightCensoredData
    install::T
end

struct LeftTruncatedRightCensoredDataset{T<:Real}
    data::Vector{LeftTruncatedRightCensoredData}
    ObservationInterval::ClosedInterval{T}
end