# Leos-Barajas, V., & Michelot, T. (2018).
# An Introduction to Animal Movement Modeling with
# Hidden Markov Models using Stan for Bayesian Inference.
# ArXiv:1806.10639 [q-Bio, Stat].
# http://arxiv.org/abs/1806.10639

using CSV
using DataFrames
using Distributions
using Random
using Turing

Random.seed!(1)

df = CSV.read("data/hmm.csv", DataFrame)

@model hmm(y, K::Int64; N=length(y)) = begin
    # state sequence
    s = tzeros(Int, N)

    # Transition Probability Matrix.
    T = Vector{Vector}(undef, K)

    # State-Dependente Parameters
    μ = Vector(undef, K)

    # Assign distributions to each element
    # of the transition matrix and the
    # state-dependent parameters.
    for i = 1:K
        T[i] ~ Dirichlet(ones(K) / K)
        μ[i] ~ TDist(3)
    end

    # Observe each point of the input.
    s[1] ~ Categorical(K)
    y[1] ~ Normal(μ[s[1]], 0.1)

    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i - 1]]))
        y[i] ~ Normal(μ[s[i]], 0.1)
    end
end

sampler = Gibbs(NUTS(1000, 0.65, :μ, :T),
                PG(50, :s))

chain = sample(hmm(df.y, 2), sampler, 2_000)
# chain = sample(hmm(df.y, 2), sampler, MCMCThreads(), 2_000, 4)
