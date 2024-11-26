module SLTRC

using Distributions
using IntervalSets
using LinearAlgebra
using Zygote

Fname(F) = typeof(F) |> nameof |> sym -> eval(Core.eval(Distributions,sym))

include("NumericalIntegration.jl")

const Default_NumericalIntegration = DETrapezoidalRule

include("Struct.jl")
include("Likelihood/Common.jl")
include("Likelihood/Exponential.jl")
include("Toolbox.jl")
include("Optimizer/Newton.jl")
include("Optimizer/MLEBase.jl")
include("Optimizer/MLEWeibull.jl")
include("Optimizer/MLEExponential.jl")
include("Sampler.jl")

export Interval
export CompleteData, RightCensoredData, StrictlyLeftTruncatedData, WeaklyLeftTruncatedData, StrictlyLeftTruncatedRightCensoredData, WeaklyLeftTruncatedRightCensoredData, LeftTruncatedRightCensoredDataset
export C, logC, ∇C, ∇²C, ∇logC, ∇ᵏlogC, ∇xC, ∇yC, ∇²xC, ∇²yC, ∇xlogC, ∇ylogC, ∇ᵏxlogC, ∇ᵏylogC
export p̃, logp̃, ∇ᵏlogp̃, ∇ᵏxlogp̃, ∇ᵏylogp̃
export loglikelihood, ∇ᵏloglikelihood, ∇ᵏxloglikelihood, ∇ᵏyloglikelihood
export conditionalloglikelihood, ∇ᵏconditionalloglikelihood
export SimpsonRule, DESimpsonRule, TrapezoidalRule, DETrapezoidalRule
export ratio, sampling
export Newton, MLE, MLE_Alternative, ConditionalMLE, RightCensoredWeibullMLE, RightCensoredExponentialMLE

end
