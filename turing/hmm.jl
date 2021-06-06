# Leos-Barajas, V., & Michelot, T. (2018).
# An Introduction to Animal Movement Modeling with
# Hidden Markov Models using Stan for Bayesian Inference.
# ArXiv:1806.10639 [q-Bio, Stat].
# http://arxiv.org/abs/1806.10639

using Distributions
using JLSO
using LinearAlgebra
using Random
using StatsBase
using Turing
using Dates:now

const seed = 1
Random.seed!(seed)

const N = 2 # Number of States

# Transition Probabilities
const Γ = Matrix([0.9 0.1; 0.1 0.9])
# initial distribution set to the stationary distribution
const δ = (Diagonal(ones(N)) - Γ .+ 1) \ ones(N)
# State-Dependent Gaussian means
const μ = [1, 5]

const nobs = 1_000
S = Vector{Int64}(undef, nobs)
y = Vector{Float64}(undef, nobs)

# initialise state and observation
S[1] = sample(1:N, aweights(δ))
y[1] = rand(Normal(μ[S[1]], 2))

# simulate state and observation processes forward
for t in 2:nobs
    S[t] = sample(1:N, aweights(Γ[S[t - 1], :]))
    y[t] = rand(Normal(μ[S[t]], 2))
end


@model hmm(y, K::Int64; T=length(y)) = begin
    # state sequence in a Libtask.TArray
    s = tzeros(Int, T)

    # Transition Probability Matrix.
    θ = Vector{Vector}(undef, K)

    # Priors
    μ ~ filldist(truncated(TDist(3), 0, Inf), 2)
    for i = 1:K
        θ[i] ~ Dirichlet(ones(K) / K)
    end

    # first observation
    s[1] ~ Categorical(K)
    y[1] ~ Normal(μ[s[1]], 2)

    # looping over observations
    for i = 2:T
        s[i] ~ Categorical(vec(θ[s[i - 1]]))
        y[i] ~ Normal(μ[s[i]], 2)
    end
end

sampler = Gibbs(NUTS(1_000, 0.65, :μ, :θ),
                PG(50, :s))

hmm_chain = sample(hmm(y, 2), sampler, MCMCThreads(), 2_000, 6)

# plot(hmm_chain)

JLSO.save("turing/hmm_chain.jlso",
          :time => now(),
          :sampler => "Gibbs(NUTS(2_000, 0.65, :μ, :θ), PG(50, :s))",
          :specs => "sample(hmm(y, 2), sampler, MCMCThreads(), 4_000, 4)",
          :seed => seed,
          :chain => hmm_chain)
