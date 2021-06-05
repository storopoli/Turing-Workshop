### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 8902a846-fbb9-42fc-8742-c9c4a84db52c
begin
	using DataFrames
	using Distributions
	using JLSO
	using LinearAlgebra
	using Random
	using StatsBase
	using StatsPlots
	using Turing
	using Plots
	using PlutoUI
	using Pkg
	#Pkg.add("TikzPictures")
	Random.seed!(1)
end

# ╔═╡ 5df4d7d2-c622-11eb-3bbd-bff9668ee5e0
md"""
# Turing Workshop
"""

# ╔═╡ 1436305e-37d8-44f1-88d6-4de838580360
md"""
## Bayesian Statistics?!
Sorry not going there... Gelman BDA and McElreath books side by side
"""

# ╔═╡ 9ebac6ba-d213-4ed8-a1d5-66b841fafa00
md"""
## Crazy Stuff
"""

# ╔═╡ d44c7baa-80d2-4fdb-a2de-35806477dd58
md"""
### Discrete Parameters (HMM)
"""

# ╔═╡ c1b2d007-1004-42f5-b65c-b4e2e7ff7d8e
PlutoUI.LocalResource("images/HMM.png", :width => 500)

# ╔═╡ f1153918-0748-4400-ae8b-3b59f8c5d755
md"""
I **love** [`Stan`](https://mc-stan.org), use it on a daily basis. But `Stan` has some quirks. Particularly, NUTS and HMC samplers **cannot tolerate discrete parameters**.

Solution? We have to **marginalize** them.

First, I will show the `Stan` example of a Hidden Markov Model (HMM) with marginalization. And then let's see how `Turing` fare with the same problem.

"""

# ╔═╡ ad6c4533-cd56-4f6f-b10d-d7bc3145ba16
md"""
We have several ways to marginalize discrete parameters in HMM:

1. **Filtering** (a.k.a [Forward Algorithm](https://en.wikipedia.org/wiki/Forward_algorithm)) <---- we'll cover this one
2. **Smoothing** (a.k.a [Forward-Backward Algorithm](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm))
3. **MAP Estimation** (a.k.a [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm))

"""

# ╔═╡ 8d347172-2d26-4d9e-954d-b8924ed4c9e2
md"""
#### Forward Algorithm

We define the forward variables $\boldsymbol{\alpha}_t$, beginning at time $t = 1$, as follows

$$\boldsymbol{\alpha}_1 = \boldsymbol{\delta}^{(1)} \boldsymbol{\Gamma}(y_1), \qquad \boldsymbol{\alpha}_{t} = \boldsymbol{\alpha}_{t-1} \boldsymbol{\Gamma} \boldsymbol{\delta}^{t-1}(y_{t})$$

where:

* Observed variable of interest at time $t$: $y_t$

* Unobserved transition probability matrix at time $t$: $\boldsymbol{\Gamma}^{(t)} \in \mathbb{R}^{j \times j}$

* Unobserved state distribution at time $t=1$: $\boldsymbol{\delta}^{(1)} \sim \text{Dirichlet}(\boldsymbol{\nu})$

Then, the marginal likelihood is obtained by summing over:

$$\sum^J_{j=1} \alpha_T (j) = \boldsymbol{\alpha}_T \mathbf{1}^{\top}$$

Note that one of the assumptions is $\boldsymbol{\Gamma}$ is **fullrank** (no linear dependence) and **ergodic** (it converges to a unique stationary distribution in $\lim_{t \to \infty}$)
"""

# ╔═╡ ca962c0e-4620-4888-b7c3-aa7f6d7899e9
md"""
As an example, I will use [Leos-Barajas & Michelot, 2018](http://arxiv.org/abs/1806.10639)'s.

It's a $2$-state HMM with Gaussian state-dependent distributions for the observation process $X_t$. That is, at each time step $(t=1,2,\dots)$, we have

$$Y_t \mid S_t = j \sim N(\mu_j,\sigma^2)$$

for $j \in \{1, 2\}$

The marginal likelihood of the model can be written with the forward algorithm using a transition matrix $\boldsymbol{\Gamma}$:

$$\boldsymbol{\Gamma}(x_t) =
\begin{pmatrix}
\phi(x_t \vert \mu_1, \sigma^2) & 0 \\
0 & \phi(x_t \vert \mu_2, \sigma^2) \\
\end{pmatrix}$$

where $\phi$ is the Gaussian PDF.

We are interested in knowing $\boldsymbol{\Gamma}$ and also $\boldsymbol{\mu}$!
"""

# ╔═╡ 6fd49295-d0e3-4b54-aeae-e9cd07a5281c
md"""
#### Random Data
"""

# ╔═╡ 58c5460f-c7f4-4a0a-9e18-71b9580e9148
begin
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
end

# ╔═╡ 6c04af9b-02af-408d-b9cb-de95ab970f83
scatter(y, mc= S, xlabel="t", ylabel="y", label=false)

# ╔═╡ 5d3d2abb-85e3-4371-926e-61ff236253f1
md"""
Here is the `Stan` code (I've simplified from Leos-Barajas & Michelot's original code) 

Also note that we are using the `log_sum_exp()` trick
"""

# ╔═╡ 247a02e5-8599-43fd-9ee5-32ba8b827477
md"""
```cpp
data {
  int<lower=1> K; // number of states
  int<lower=1> T; // length of data set
  real y[T]; // observations
}
parameters {
  positive_ordered[K] mu; // state-dependent parameters
  simplex[K] theta[K]; // N x N tpm
}
model{
  // priors
  mu ~ student_t(3, 0, 1);
  for (k in 1:K)
    theta[k] ~ dirichlet([0.5, 0.5]);

  // Compute the marginal probability over possible sequences
  vector[K] acc;
  vector[K] lp;
  // forward algorithm implementation
  for(k in 1:K) // first observation
    lp[k] = normal_lpdf(y[1] | mu[k], 2);
  for (t in 2:T) {     // looping over observations
      for (k in 1:K){   // looping over states
          acc[k] = log_sum_exp(log(theta[k]) + lp) +
            normal_lpdf(y[t] | mu[k], 2);
      }
      lp = acc;
    }
  target += log_sum_exp(lp);
}
```
"""

# ╔═╡ 6db0245b-0461-4db0-9462-7a5f80f7d589
md"""
Here's how we would do in Turing

Note the Composite MCMC Sampler

```julia
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

hmm_chain = sample(hmm(y, 2), sampler, MCMCThreads(), 2_000, 4)
```
"""

# ╔═╡ 9b0b62cb-2c61-4d47-a6c7-09c0c1a75a24
md"""
## ODEs in `Turing` (SIR Model)
"""

# ╔═╡ 9b020402-ea15-4f52-9fff-c70d275b97ac
PlutoUI.LocalResource("images/SIR.png", :width => 500)

# ╔═╡ c81f4877-024f-4dc8-b7ce-e781ab6101f3
md"""
The Susceptible-Infected-Recovered (SIR) model splits the population in three time-dependent compartments: the susceptible, the infected (and infectious), and the recovered (and not infectious) compartments. When a susceptible individual comes into contact with an infectious individual, the former can become infected for some time, and then recover and become immune. The dynamics can be summarized in a system ODEs:
"""

# ╔═╡ f2272fd5-5132-4a6e-b2ff-136dc2fb2903
md"""
$$\begin{aligned}
 \frac{dS}{dt} &= -\beta  S \frac{I}{N}\\
 \frac{dI}{dt} &= \beta  S  \frac{I}{N} - \gamma  I \\
 \frac{dR}{dt} &= \gamma I
\end{aligned}$$

where

*  $S(t)$ is the number of people susceptible to becoming infected (no immunity),

*  $I(t)$ is the number of people currently infected (and infectious),

*  $R(t)$ is the number of recovered people (we assume they remain immune indefinitely),

*  $\beta$ is the constant rate of infectious contact between people,

*  $\gamma$ the constant recovery rate of infected individuals.

"""

# ╔═╡ 5c017766-445d-4f4b-98f1-ae63e78ec34b
md"""
As an example, I will use [Grinsztajn, Semenova, Margossian & Riou. 2021)](https://arxiv.org/abs/2006.02985)'s.

It's a boarding school:
"""

# ╔═╡ 2f907e0d-171e-44c3-a531-5f11da08b3cf
md"""
## Pluto Stuff
"""

# ╔═╡ 31b6d4ec-d057-44ca-875b-0c3257895dd3
PlutoUI.TableOfContents(aside=true)

# ╔═╡ 3c8a9b24-9863-42bb-974c-6b2f46134567
# The HMM model takes a while to run so I've pre-ran and loaded as a JLSO file
begin
	loaded = JLSO.load("turing/hmm_chain.jlso")
	hmm_chain = loaded[:chain]
end

# ╔═╡ 8d2487e9-47c0-4202-abc5-34f5e0a18f74
sum(S .== quantile(group(hmm_chain, :s))[:, :var"50.0%"])

# ╔═╡ 8c768ea7-01ad-4eea-bdbd-042c58d843aa
summarystats(cat(group(hmm_chain, :μ), group(hmm_chain, :θ); dims=2))

# ╔═╡ 98ece9fe-dfcc-4dd8-bd47-049217d2afcf
md"""
## References

Grinsztajn, L., Semenova, E., Margossian, C. C., & Riou, J. (2021). Bayesian workflow for disease transmission modeling in Stan. ArXiv:2006.02985 [q-Bio, Stat]. http://arxiv.org/abs/2006.02985


Leos-Barajas, V., & Michelot, T. (2018). An Introduction to Animal Movement Modeling with Hidden Markov Models using Stan for Bayesian Inference. ArXiv:1806.10639 [q-Bio, Stat]. http://arxiv.org/abs/1806.10639
"""

# ╔═╡ Cell order:
# ╟─5df4d7d2-c622-11eb-3bbd-bff9668ee5e0
# ╠═1436305e-37d8-44f1-88d6-4de838580360
# ╟─9ebac6ba-d213-4ed8-a1d5-66b841fafa00
# ╟─d44c7baa-80d2-4fdb-a2de-35806477dd58
# ╟─c1b2d007-1004-42f5-b65c-b4e2e7ff7d8e
# ╟─f1153918-0748-4400-ae8b-3b59f8c5d755
# ╟─ad6c4533-cd56-4f6f-b10d-d7bc3145ba16
# ╟─8d347172-2d26-4d9e-954d-b8924ed4c9e2
# ╟─ca962c0e-4620-4888-b7c3-aa7f6d7899e9
# ╟─6fd49295-d0e3-4b54-aeae-e9cd07a5281c
# ╠═58c5460f-c7f4-4a0a-9e18-71b9580e9148
# ╠═6c04af9b-02af-408d-b9cb-de95ab970f83
# ╟─5d3d2abb-85e3-4371-926e-61ff236253f1
# ╟─247a02e5-8599-43fd-9ee5-32ba8b827477
# ╟─6db0245b-0461-4db0-9462-7a5f80f7d589
# ╠═8d2487e9-47c0-4202-abc5-34f5e0a18f74
# ╠═8c768ea7-01ad-4eea-bdbd-042c58d843aa
# ╟─9b0b62cb-2c61-4d47-a6c7-09c0c1a75a24
# ╟─9b020402-ea15-4f52-9fff-c70d275b97ac
# ╟─c81f4877-024f-4dc8-b7ce-e781ab6101f3
# ╟─f2272fd5-5132-4a6e-b2ff-136dc2fb2903
# ╟─5c017766-445d-4f4b-98f1-ae63e78ec34b
# ╟─2f907e0d-171e-44c3-a531-5f11da08b3cf
# ╠═31b6d4ec-d057-44ca-875b-0c3257895dd3
# ╠═8902a846-fbb9-42fc-8742-c9c4a84db52c
# ╠═3c8a9b24-9863-42bb-974c-6b2f46134567
# ╟─98ece9fe-dfcc-4dd8-bd47-049217d2afcf
