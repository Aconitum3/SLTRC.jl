module SLTRC

using Distributions
using IntervalSets
using LinearAlgebra
using SpecialFunctions
using Zygote

Fname(F) = typeof(F) |> nameof |> sym -> eval(Core.eval(Distributions,sym))

include("NumericalIntegration.jl")

const Default_NumericalIntegration = DETrapezoidalRule

include("Struct.jl")
include("Likelihood/Common.jl")
include("Likelihood/Exponential.jl")
include("Likelihood/Uniform.jl")
include("Likelihood/ImproperUniform.jl")
include("FisherInformation.jl")
include("Toolbox.jl")
include("Optimizer/Newton.jl")
include("Optimizer/MLEBase.jl")
include("Optimizer/MLEWeibull.jl")
include("Optimizer/MLEExponential.jl")
include("Optimizer/MLEUniform.jl")
include("Sampler.jl")

export Interval
export CompleteData, RightCensoredData, StrictlyLeftTruncatedData, WeaklyLeftTruncatedData, StrictlyLeftTruncatedRightCensoredData, WeaklyLeftTruncatedRightCensoredData, LeftTruncatedRightCensoredDataset
export C, logC, ∇C, ∇²C, ∇logC, ∇ᵏlogC, ∇xC, ∇yC, ∇²xC, ∇²yC, ∇xlogC, ∇ylogC, ∇ᵏxlogC, ∇ᵏylogC
export p̃, logp̃, ∇ᵏlogp̃, ∇ᵏxlogp̃, ∇ᵏylogp̃
export loglikelihood, ∇ᵏloglikelihood, ∇ᵏxloglikelihood, ∇ᵏyloglikelihood
export ImproperUniform
export conditionalloglikelihood, ∇ᵏconditionalloglikelihood
export FisherInformation, FisherInformation_CLM, FisherInformation_Allison, FisherInformationMC
export SimpsonRule, DESimpsonRule, TrapezoidalRule, DETrapezoidalRule
export ratio, sampling
export Newton, MLE, MLE_Alternative, ConditionalMLE, RightCensoredWeibullMLE, RightCensoredExponentialMLE

end
