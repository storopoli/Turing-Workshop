# Example taken from
# https://elevanth.org/blog/2018/01/29/algebra-and-missingness/
using CSV
using DataFrames
using Distributions
using Random
using StatsBase
using Turing

const N_children = 51

Random.seed!(1)
# s = rand(Binomial(1, 0.75), N_children)
# s_obs = s
# s_obs[sample(1:N_children, 21)] .= -1
# tea = rand.(Binomial.(1, s .* 1 .+ (1 .- s) .* 0.5))

df = CSV.read("data/tea.csv", DataFrame)

@model hmm(tea, s) = begin
    # Get observation length.
    p_cheat ~ Beta(2, 2)
    p_drink ~ Beta(2, 2)
    sigma ~ Beta(2, 2)

    # probability of tea
    for i in 1:length(tea)
        if s[i] == -1 # ox unobserved
            tea[i] ~ Bernoulli(p_drink * sigma + (1 - sigma) * p_cheat)
        else         # ox observed
            tea[i] ~ Bernoulli(s[i] * p_drink + (1 - s[i]) * p_cheat)
            s[i] ~ Bernoulli(sigma)
        end
    end
end

sampler = Gibbs(NUTS(1000, 0.65, :p_cheat, :p_drink, :sigma),
                PG(50, :s))
chain = sample(hmm(df.tea, df.s_obs), sampler, MCMCThreads(), 2_000, 4)

summarystats(chain)
