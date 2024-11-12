module SLTRC

using Distributions
using IntervalSets
using LinearAlgebra
using Zygote

Fname(F) = typeof(F) |> nameof |> sym -> eval(Core.eval(Distributions,sym))

include("NumericalIntegration.jl")

Default_NumericalIntegration = DETrapezoidalRule

include("Struct.jl")
include("Likelihood/Common.jl")
include("Likelihood/Exponential.jl")
include("Toolbox.jl")
include("Optimizer.jl")
include("Sampler.jl")

export CompleteData, RightCensoredData, StrictlyLeftTruncatedData, WeaklyLeftTruncatedData, StrictlyLeftTruncatedRightCensoredData, WeaklyLeftTruncatedRightCensoredData, LeftTruncatedRightCensoredDataset
export C, logC, ∇C, ∇²C, ∇logC, ∇ᵏlogC, ∇xC, ∇yC, ∇²xC, ∇²yC, ∇xlogC, ∇ylogC, ∇ᵏxlogC, ∇ᵏylogC
export p̃, logp̃, ∇ᵏlogp̃, ∇ᵏxlogp̃, ∇ᵏylogp̃
export ∇ᵏloglikelihood, ∇ᵏxloglikelihood, ∇ᵏyloglikelihood
export SimpsonRule, DESimpsonRule
export ratio, sampling
export Newton, MLE, MLE_Alternative

end
