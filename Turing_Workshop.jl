### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 8902a846-fbb9-42fc-8742-c9c4a84db52c
begin
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="CSV", version="0.8.5"),
        Pkg.PackageSpec(name="Chain", version="0.4.6"),
        Pkg.PackageSpec(name="DataFrames", version="1.1.1"),
        Pkg.PackageSpec(name="DifferentialEquations", version="6.17.1"),
        Pkg.PackageSpec(name="Distributions", version="0.24.18"),
		Pkg.PackageSpec(name="LaTeXStrings", version="1.2.1"),
		Pkg.PackageSpec(name="LazyArrays", version="0.21.5"),
		Pkg.PackageSpec(name="Plots", version="1.16.2"),
		Pkg.PackageSpec(name="PlutoUI", version="0.7.9"),
		Pkg.PackageSpec(name="StatsBase", version="0.33.8"),
		Pkg.PackageSpec(name="StatsPlots", version="0.14.21"),
		Pkg.PackageSpec(name="Turing", version="0.16.0"),
		
    ])
	using CSV
	using DataFrames
	using DifferentialEquations
	using Distributions
	using LaTeXStrings
	using LinearAlgebra
	using Random
	using StatsBase
	using StatsPlots
	using Turing
	using Plots
	using PlutoUI
end

# ╔═╡ 31161289-1d4c-46ba-8bd9-e687fb7da29e
begin
	using InteractiveUtils
	with_terminal() do
		versioninfo()
	end
end

# ╔═╡ 5df4d7d2-c622-11eb-3bbd-bff9668ee5e0
md"""
# Turing Workshop
"""

# ╔═╡ dceb8312-230f-4e4b-9285-4e23f219b838
Resource("https://github.com/storopoli/Turing-Workshop/blob/master/images/bayes-meme.jpg?raw=true", :width => 250, :align=>"center")

# ╔═╡ cda7dc96-d983-4e31-9298-6148205b54b1
md"""
A little bit about myself:

$(Resource("https://github.com/storopoli/Turing-Workshop/blob/master/images/profile_pic.jpg?raw=true", :width => 100, :align => "right"))

* **Jose Storopoli**, PhD
* Associate Professor at **Universidade Nove de Julho** (UNINOVE)
* Teach undergraduates **Statistics** and **Machine Learning** (using Python 😓)
* Teach graduate students **Bayesian Statistics** (using `Stan`) and **Scientific Computing** (using **Julia** 🚀)
* I've made some `Turing` tutorials, you can check them out at [storopoli.io/Bayesian-Julia](https://storopoli.io/Bayesian-Julia)
* You can find me on [Twitter](https://twitter.com/JoseStoropoli) (altough I've rarelly use it) or on [LinkedIn](https://www.linkedin.com/in/storopoli/)
"""

# ╔═╡ 1436305e-37d8-44f1-88d6-4de838580360
md"""
## Bayesian Statistics?!

**Bayesian statistics** is an approach to inferential statistics based on Bayes' theorem, where available knowledge about parameters in a statistical model is updated with the information in observed data. The background knowledge is expressed as a prior distribution and combined with observational data in the form of a likelihood function to determine the posterior distribution. The posterior can also be used for making predictions about future events.

$$\underbrace{P(\theta \mid y)}_{\text{Posterior}} = \frac{\overbrace{P(y \mid  \theta)}^{\text{Likelihood}} \cdot \overbrace{P(\theta)}^{\text{Prior}}}{\underbrace{P(y)}_{\text{Normalizing Costant}}}$$

> No $p$-values! Nobody knows what they are anyway... Not $P(H_0 \mid y)$
"""

# ╔═╡ 08f508c4-233a-4bba-b313-b04c1d6c4a4c
md"""
### Recommended Books
"""

# ╔═╡ 868d8932-b108-41d9-b4e8-d62d31b5465d
md"""
We are not covering Bayesian stuff, but there are some **awesome books**:
"""

# ╔═╡ 653ec420-8de5-407e-91a9-f045e25a6395
md"""
[$(Resource("https://github.com/storopoli/Turing-Workshop/blob/master/images/BDA_book.jpg?raw=true", :width => 100.5*1.5))](https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955)
[$(Resource("https://github.com/storopoli/Turing-Workshop/blob/master/images/SR_book.jpg?raw=true", :width => 104*1.5))](https://www.routledge.com/Statistical-Rethinking-A-Bayesian-Course-with-Examples-in-R-and-STAN/McElreath/p/book/9780367139919)
[$(Resource("https://github.com/storopoli/Turing-Workshop/blob/master/images/ROS_book.jpg?raw=true", :width => 118*1.5))](https://www.cambridge.org/fi/academic/subjects/statistics-probability/statistical-theory-and-methods/regression-and-other-stories)
[$(Resource("https://github.com/storopoli/Turing-Workshop/blob/master/images/Bayes_book.jpg?raw=true", :width => 102*1.5))](https://www.amazon.com/Theory-That-Would-Not-Die/dp/0300188226/)
"""

# ╔═╡ 716cea7d-d771-46e9-ad81-687292004009
md"""
## 1. What is Turing?
"""

# ╔═╡ cb808fd4-6eb2-457e-afa4-58ae1be09aec
md"""
[**`Turing`** (Ge, Xu & Ghahramani, 2018)](http://turing.ml/) is a ecosystem of Julia packages for Bayesian Inference using [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming). Models specified using `Turing` are easy to read and write -- models work the way you write them. Like everything in Julia, `Turing` is fast [(Tarek, Xu, Trapp, Ge & Ghahramani, 2020)](https://arxiv.org/abs/2002.02702).

Before we dive into how to specify models in Turing. Let's discuss Turing's **ecosystem**.
We have several Julia packages under the Turing's GitHub organization [TuringLang](https://github.com/TuringLang), but I will focus on 5 of those:

* [`Turing.jl`](https://github.com/TuringLang/Turing.jl): main package that we use to **interface with all the Turing ecosystem** of packages and the backbone of the PPL Turing.

* [`MCMCChains.jl`](https://github.com/TuringLang/MCMCChains.jl): is an interface to **summarizing MCMC simulations** and has several utility functions for **diagnostics** and **visualizations**.

* [`DynamicPPL.jl`](https://github.com/TuringLang/DynamicPPL.jl): which specifies a domain-specific language and backend for Turing (which itself is a PPL), modular and written in Julia

* [`AdvancedHMC.jl`](https://github.com/TuringLang/AdvancedHMC.jl): modular and efficient implementation of advanced HMC algorithms. The state-of-the-art HMC algorithm is the **N**o-**U**-**T**urn **S**ampling (NUTS) (Hoffman & Gelman, 2011)

* [`DistributionsAD.jl`](https://github.com/TuringLang/DistributionsAD.jl): defines the necessary functions to enable automatic differentiation (AD) of the `logpdf` function from [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) using the packages [`Tracker.jl`](https://github.com/FluxML/Tracker.jl), [`Zygote.jl`](https://github.com/FluxML/Zygote.jl), [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) and [`ReverseDiff.jl`](https://github.com/JuliaDiff/ReverseDiff.jl). The main goal of `DistributionsAD.jl` is to make the output of `logpdf` differentiable with respect to all continuous parameters of a distribution.
"""

# ╔═╡ 0484ae7f-bd8a-4615-a760-5c4b2eef9d3f
md"""
## 2. How to Specify a Model? `@model`
"""

# ╔═╡ 1d467044-bc7d-4df7-bda6-bb8ea6ff0712
md"""
**We specify the model inside a macro** `@model` where we can assign variables in two ways:

* using `~`: which means that a variable follows some probability distribution (Normal, Binomial etc.) and its value is random under that distribution

* using `=`: which means that a variable does not follow a probability distribution and its value is deterministic (like the normal `=` assignment in programming languages)

Turing will perform automatic inference on all variables that you specify using `~`.

> **Example**: Unfair coin with $p$ = 0.7.
"""

# ╔═╡ b1d99482-53f5-4c6b-8c20-c761ff6bdb77
coin_flips = rand(Bernoulli(0.7), 100)

# ╔═╡ 9f6b96a7-033d-4c7d-a853-46a0b5af4675
md"""
## 3. How to specify a MCMC sampler (`NUTS`, `HMC`, `MH` etc.)
"""

# ╔═╡ b7667fb4-6e76-4711-b61d-dae5f993531e
md"""
We have [several samplers](https://turing.ml/dev/docs/using-turing/sampler-viz) available:

* `MH()`: **M**etropolis-**H**astings
* `PG()`: **P**article **G**ibbs
* `HMC()`: **H**amiltonian **M**onte **C**arlo
* `HMCDA()`: **H**amiltonian **M**onte **C**arlo with Nesterov's **D**ual **A**veraging
* `NUTS()`: **N**o-**U**-**T**urn **S**ampling

Just stick your desired `sampler` inside the function `sample(model, sampler, N; kwargs)`.

Play around if you want. Choose your `sampler`:
"""

# ╔═╡ cb168dc1-70e2-450f-b2cf-c8680251ab27
@bind chosen_sampler Radio(["MH", "PG", "HMC", "HMCDA", "NUTS"], default = "MH")

# ╔═╡ 07d408cf-d202-40b2-90c2-5e8630549339
begin
	your_sampler = nothing
	if chosen_sampler == "MH"
		your_sampler = MH()
	elseif chosen_sampler == "PG"
		your_sampler = PG(2)
	elseif chosen_sampler == "HMC"
		your_sampler = HMC(0.05, 10)
	elseif chosen_sampler == "HMCDA"
		your_sampler = HMCDA(10, 0.65, 0.3)
	elseif chosen_sampler == "NUTS"
		your_sampler = NUTS(10, 0.65)
	end
end

# ╔═╡ e6365296-cd68-430e-99c5-fb571f39aad5
md"""
### 3.1 MOAH CHAINS!!: `MCMCThreads` and `MCMCDistributed`
"""

# ╔═╡ 927ad0a4-ba68-45a6-9bde-561915503e48
md"""
The difference between `MCMCThreads()` and `MCMCDistributed()` ...
"""

# ╔═╡ 2ab3c34a-1cfc-4d20-becc-5902d08d03e0
md"""
### 3.2 LOOK MUM NO DATA!!: Prior Predictive Checks `Prior()`
"""

# ╔═╡ 924fcad9-75c1-4707-90ef-3e36947d64fe
md"""
It's very important that we check if our **priors make sense**. This is called **Prior Predictive Check** (Gelman et al., 2020b). Obs: I will not cover **Posterior Predictive Check** because is mostly the same procedure in `Turing`.
"""

# ╔═╡ fc8e40c3-34a1-4b2e-bd1b-893d7998d359
md"""
$(Resource("https://github.com/storopoli/Turing-Workshop/blob/master/images/bayesian_workflow.png?raw=true", :width => 700))

Based on Gelman et al. (2020b)
"""

# ╔═╡ fb366eb1-4ab0-4e7a-83ed-d531978c06a0
md"""
To do...
"""

# ╔═╡ 5674f7aa-3205-47c7-8367-244c6419ce69
md"""
## 4. How to inspect chains and plot stuff with `MCMCChains.jl`
"""

# ╔═╡ 83cc80c1-d97e-4b82-872e-e5493d2b62ab
md"""
To do...
"""

# ╔═╡ c70ebb70-bd96-44a5-85e9-871b0e478b1a
md"""
## 5. Better tricks to avoid for-loops inside `@model` (`lazyarrays` and `filldist`)
"""

# ╔═╡ dd5fbb2a-4220-4e47-945a-6870b799c50d
md"""
To do...
"""

# ╔═╡ 7d4d06ca-f96d-4b1e-860f-d9e0d6eb6723
md"""
## 6. Take me up! Let's get Hierarchical (Hierarchical Models)
"""

# ╔═╡ c64d355f-f5a2-46a5-86f3-2d02da98f305
md"""
To do...
"""

# ╔═╡ 9ebac6ba-d213-4ed8-a1d5-66b841fafa00
md"""
## 7. Crazy Stuff
"""

# ╔═╡ 45c342fd-b893-46aa-b2ee-7c93e7a1d207
md"""
To do...
"""

# ╔═╡ d44c7baa-80d2-4fdb-a2de-35806477dd58
md"""
### 7.1 Discrete Parameters (HMM)
"""

# ╔═╡ c1b2d007-1004-42f5-b65c-b4e2e7ff7d8e
Resource("https://github.com/storopoli/Turing-Workshop/blob/master/images/HMM.png?raw=true", :width => 400)

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

A very good reference is [Damiano, Peterson & Weylandt (2017)](https://github.com/luisdamiano/stancon18)

"""

# ╔═╡ 2ef397a6-f7fb-4fc2-b918-40ab545ce19f
md"""
#### Forward Algorithm
"""

# ╔═╡ 8d347172-2d26-4d9e-954d-b8924ed4c9e2
md"""
We define the forward variables $\boldsymbol{\alpha}_t$, beginning at time $t = 1$, as follows

$$\boldsymbol{\alpha}_1 = \boldsymbol{\delta}^{(1)} \boldsymbol{\Gamma}(y_1), \qquad \boldsymbol{\alpha}_{t} = \boldsymbol{\alpha}_{t-1} \boldsymbol{\Gamma} \boldsymbol{\delta}^{t-1}(y_{t})$$

where:

* Observed variable of interest at time $t$: $y_t$

* Unobserved transition probability matrix at time $t$: $\boldsymbol{\Gamma}^{(t)} \in \mathbb{R}^{j \times j}$

* Unobserved state distribution at time $t=1$: $\boldsymbol{\delta}^{(1)} \sim \text{Dirichlet}(\boldsymbol{\nu})$

Then, the marginal likelihood is obtained by summing over:

$$\sum^J_{j=1} \alpha_T (j) = \boldsymbol{\alpha}_T \mathbf{1}^{\top}$$

(dot product of the vector of $\alpha$s with a row vector of $1$s)

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
begin
	scatter(y, mc= S, xlabel=L"t", ylabel=L"y", label=false)
	hline!([1,5], lw=4, label=false, c=:black, style=:dash)
end

# ╔═╡ 5d3d2abb-85e3-4371-926e-61ff236253f1
md"""
Here is the `Stan` code (I've simplified from Leos-Barajas & Michelot's original code) :

> Also note that we are using the `log_sum_exp()` trick
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

Obs: `log_sum_exp(a, b) = log(exp(a) + exp(b))`
"""

# ╔═╡ 6db0245b-0461-4db0-9462-7a5f80f7d589
md"""
Here's how we would do in `Turing`

> Note the Composite MCMC Sampler 
"""

# ╔═╡ b5a79826-151e-416e-b0a2-1a58eec9196c
begin
	@model hmm(y, K::Int64; T=length(y)) = begin
		# state sequence in a Libtask.TArray
		s = tzeros(Int, T)

		# Transition Probability Matrix.
		θ = Vector{Vector}(undef, K)

		# Priors
		μ ~ filldist(truncated(TDist(3), 1, 6), 2)
		for i = 1:K
			θ[i] ~ Dirichlet([0.5, 0.5])
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

	sampler = Gibbs(NUTS(10, 0.65, :μ, :θ),
					PG(1, :s))

	hmm_chain = sample(hmm(y, 2), sampler, 50)
	summarystats(hmm_chain[:, 1:6, :]) #only μ and θ
end

# ╔═╡ 9b0b62cb-2c61-4d47-a6c7-09c0c1a75a24
md"""
### 7.2 ODEs in `Turing` (SIR Model)
"""

# ╔═╡ 9b020402-ea15-4f52-9fff-c70d275b97ac
Resource("https://github.com/storopoli/Turing-Workshop/blob/master/images/SIR.png?raw=true", :width => 400)

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

# ╔═╡ 2d230fea-dcf2-41e6-a477-2a2334f56990
md"""
#### How to code and ODE in Julia?

It's very easy:

1. Use [`DifferentialEquations.jl`](https://diffeq.sciml.ai/)
2. Create a ODE function
3. Choose:
   * Initial Conditions: $u_0$
   * Parameters: $p$
   * Time Span: $t$
   * *Optional*: [Solver](https://diffeq.sciml.ai/stable/solvers/ode_solve/) or leave blank for auto

PS: If you like SIR models checkout [`epirecipes/sir-julia`](https://github.com/epirecipes/sir-julia)
"""

# ╔═╡ 44f9935f-c5a5-4f08-a94b-7f6ee70df358
function sir_ode!(du, u, p, t)
	    (S, I, R) = u
	    (β, γ) = p
	    N = S + I + R
	    infection = β * I / N * S
	    recovery = γ * I
	    @inbounds begin
	        du[1] = -infection
	        du[2] = infection - recovery
	        du[3] = recovery
	    end
	    nothing
	end;

# ╔═╡ 92e17d42-c6d1-4891-99a9-4a3be9e2decf
md"""
I₀ = $(@bind I₀ Slider(1:1:20, default = 1, show_value=true))

β = $(@bind sim_β Slider(0.1:0.2:3, default = 1.9, show_value=true))

γ = $(@bind sim_γ Slider(0.1:0.1:1.5, default = 0.9, show_value=true))
"""

# ╔═╡ 39902541-5243-4fa9-896c-36db93d9fcea
begin
	u = [763, I₀, 0]
	p = [sim_β, sim_γ]
	tspan = (0.0, 20.0)
end

# ╔═╡ 65fa382d-4ef7-432d-8630-27082977185b
@model coin(coin_flips) = begin
	p ~ Beta(1,1)
	for i in 1:length(coin_flips)
		coin_flips[i] ~ Bernoulli(p)
	end
end


# ╔═╡ 06f93734-2315-4b36-a39a-09e8167bab1f
begin
	chain_coin = sample(coin(coin_flips), MH(), 100)
	summarystats(chain_coin)
end

# ╔═╡ 744a8a63-647f-4550-adf7-44354fde44be
begin
	chain_coin_2 = sample(coin(coin_flips), your_sampler, 100) # Here is your sampler
	summarystats(chain_coin_2)
end

# ╔═╡ 646ab8dc-db5a-4eb8-a08b-217c2f6d86be
begin
	Plots.gr()
	prob = ODEProblem(sir_ode!, u, tspan, p)
	sol = solve(prob, Tsit5(), saveat=1.0)
	plot(sol, dpi=300, label=[L"S" L"I" L"R"], lw=3)
	xlabel!("days")
	ylabel!("N")
end

# ╔═╡ 5c017766-445d-4f4b-98f1-ae63e78ec34b
md"""
As an example, I will use [Grinsztajn, Semenova, Margossian & Riou. 2021)](https://arxiv.org/abs/2006.02985)'s.

It's a boarding school:

> Outbreak of **influenza A (H1N1)** in 1978 at a British boarding school. The data consists of the daily number of students in bed, spanning over a time interval of 14 days. There were **763 male students** who were mostly full boarders and 512 of them became ill.  The outbreak lasted from the 22nd of January to the 4th of February. It is reported that **one infected boy started the epidemic**, which spread rapidly in the relatively closed community of the boarding school.

The data are freely available in the R package `{outbreaks}`, maintained as part of the [R Epidemics Consortium](http://www.repidemicsconsortium.org).
"""

# ╔═╡ 0a76f019-4853-4ba3-9af8-9f33e1d4c956
begin
	# Boarding School SIR
	boarding_school = CSV.read(download("https://github.com/storopoli/Turing-Workshop/blob/master/data/influenza_england_1978_school.csv?raw=true"), DataFrame)
	cases = boarding_school.in_bed
end

# ╔═╡ 680f104e-80b4-443f-b4bc-532df758c162
md"""
Here's how we would do in `Turing`:

> Note the ODE system inside `@model`
"""

# ╔═╡ ddfc38fc-b47d-4ea5-847a-e9cbee3aa0a1
@model sir(cases, I₀) = begin
  # Calculate number of timepoints
  l = length(cases)
  N = 763
  S₀ = N - I₀
  R₀ = 0

  # Priors
  β ~ TruncatedNormal(2, 1,  1e-6, 10)     # using 10 instead of `Inf` because numerical issues arose
  γ ~ TruncatedNormal(0.4, 0.5,  1e-6, 10) # using 10 instead of `Inf` because numerical issues arose
  ϕ⁻ ~ truncated(Exponential(5), 1, 20)
  ϕ = 1.0 / ϕ⁻

  # ODE Stuff
  u = float.([S₀, I₀, R₀])
  p = [β, γ]
  tspan = (0.0, float(l))
  prob = ODEProblem(sir_ode!,
          u,
          tspan,
          p)
  sol = solve(prob,
              Tsit5(), # You can change the solver (similar to RK45)
              saveat=1.0)
  solᵢ = Array(sol)[2, 2:end] # Infected

  # Likelihood
  for i in 1:l
    solᵢ[i] = max(1e-6, solᵢ[i]) # numerical issues arose
    cases[i] ~ NegativeBinomial(solᵢ[i], ϕ)
  end
end

# ╔═╡ ee2616ca-2602-4823-9cfb-123b958701c4
begin
	sir_chain = sample(sir(cases, 1), NUTS(1_000, 0.65), MCMCThreads(), 2_000, 2)
	summarystats(sir_chain[:, 1:2, :]) # only β and γ
end

# ╔═╡ 7a62c034-3709-483a-a663-7fe5e09cb773
begin
	Plots.gr()
	plot(sir_chain[:, 1:2, :]) # only β and γ
end

# ╔═╡ 7f1fd9b4-517a-4fec-89bb-4d696dadbc3d
md"""
## 8.1 Computational Tricks
"""

# ╔═╡ 81e29fc7-b5d3-46d8-aeac-fb8e6dc11b16
md"""
### 8.1 Non-centered parametrization (Funnel of Death)
"""

# ╔═╡ 5291b260-9a68-4c8b-aff4-7797804ccc95
md"""
Sometimes our posterior has **crazy geometries** that makes our MCMC sampler (including NUTS and HMC) to have a hard time to sample from it.

This example is from Neal (2003) and is called Neal's Funnel (altough some call it Funnel of Death). It exemplifies the difficulties of sampling from some hierarchical models. Here I will show a 2-D example with $x$ and $y$:

$$p(y,x) = \text{Normal}(y \mid 0,3) \times
\text{normal}\left(x \mid 0,\exp\left(\frac{y}{2}\right)\right)$$
"""

# ╔═╡ fe0fefb6-2755-4319-a944-bbbc7843aead
begin
	Plots.plotly()
	x = -2:0.01:2;
	kernel(x, y) = logpdf(Normal(0, exp(y / 2)), x)
	surface(x, x, kernel, xlab="x", ylab="y", zlab="log(PDF)")
end

# ╔═╡ c109b759-7b73-4593-b9ea-8cc97b61d6fe
md"""
#### Whats the problem here?

* At the *bottom* of the funnel: **low** $\epsilon$ and **high** $L$
* At the *top* of the funnel: **high** $\epsilon$ and **low** $L$

HMC you have to set your $\epsilon$ and $L$ so it's fixed.

NUTS can automatically set $\epsilon$ and $L$ during warmup (it can vary) but it's fixed during sampling.

So basically you are screwed if you do not reparametrize!
"""

# ╔═╡ 60494b7c-1a08-4846-8a80-12533552a697
md"""
#### Reparametrization
What if we reparameterize so that we can express $y$ and $x$ as standard normal distributions, by using a reparameterization trick:

$$\begin{aligned}
x^* &\sim \text{Normal}(0, 1)\\
x &= x^* \cdot \sigma_x + \mu_x
\end{aligned}$$

This also works for multivariate stuff
"""

# ╔═╡ b57195f9-c2a1-4676-96f9-faee84f7fc26
md"""
#### Non-Centered Reparametrization of the Funnel of Death

We can provide the MCMC sampler a better-behaved posterior geometry to explore:


$$\begin{aligned}
p(y^*,x^*) &= \text{Normal}(y^* \mid 0,0) \times
\text{Normal}(x^* \mid 0,0)\\
y &= 3y^* + 0\\
x &= \exp \left( \frac{y}{2} \right) x^* + 0
\end{aligned}$$

Below there is is the Neal's Funnel reparameterized as standard normal:
"""

# ╔═╡ 438d437e-7b00-4a13-8f8a-87fdc332a190
begin
	Plots.plotly()
	kernel_reparameterized(x, y) = logpdf(Normal(), x)
	surface(x, x,  kernel_reparameterized, xlab="x", ylab="y", zlab="log(PDF)")
end

# ╔═╡ 26265a91-2c8e-46d8-9a87-a2d097e7433a
md"""
### 8.2 $\mathbf{QR}$ decomposition
"""

# ╔═╡ 2eeb402e-c5f9-449c-af19-ff8f2e6c7246
md"""
To do...
"""

# ╔═╡ 2f907e0d-171e-44c3-a531-5f11da08b3cf
md"""
## Pluto Stuff
"""

# ╔═╡ 31b6d4ec-d057-44ca-875b-0c3257895dd3
PlutoUI.TableOfContents(aside=true)

# ╔═╡ b75f8003-85d4-4bb7-96cf-b6d7881b0e7c
md"""
## Backup Computations
"""

# ╔═╡ 4af78efd-d484-4241-9d3c-97cc78e1dbd4
Random.seed!(1)

# ╔═╡ 98ece9fe-dfcc-4dd8-bd47-049217d2afcf
md"""
## References

Damiano, L., Peterson, B., & Weylandt, M. (2017). A Tutorial on Hidden Markov Models using Stan. https://github.com/luisdamiano/stancon18 (Original work published 2017)

Ge, H., Xu, K., & Ghahramani, Z. (2018). Turing: A Language for Flexible Probabilistic Inference. International Conference on Artificial Intelligence and Statistics, 1682–1690. http://proceedings.mlr.press/v84/ge18b.html

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis*. Chapman and Hall/CRC.

Gelman, A., Hill, J., & Vehtari, A. (2020a). *Regression and other stories*. Cambridge University Press.

Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., & Modrák, M. (2020b). Bayesian Workflow. ArXiv:2011.01808 [Stat]. http://arxiv.org/abs/2011.01808

Grinsztajn, L., Semenova, E., Margossian, C. C., & Riou, J. (2021). Bayesian workflow for disease transmission modeling in Stan. ArXiv:2006.02985 [q-Bio, Stat]. http://arxiv.org/abs/2006.02985

Hoffman, M. D., & Gelman, A. (2011). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593–1623.

Leos-Barajas, V., & Michelot, T. (2018). An Introduction to Animal Movement Modeling with Hidden Markov Models using Stan for Bayesian Inference. ArXiv:1806.10639 [q-Bio, Stat]. http://arxiv.org/abs/1806.10639

McElreath, R. (2020). *Statistical rethinking: A Bayesian course with examples in R and Stan*. CRC press.

McGrayne, S.B (2012). *The Theory That Would Not Die: How Bayes' Rule Cracked the Enigma Code, Hunted Down Russian Submarines, and Emerged Triumphant from Two Centuries of Controversy* Yale University Press.

Neal, R. M. (2003). Slice Sampling. The Annals of Statistics, 31(3), 705–741.

Tarek, M., Xu, K., Trapp, M., Ge, H., & Ghahramani, Z. (2020). DynamicPPL: Stan-like Speed for Dynamic Probabilistic Models. ArXiv:2002.02702 [Cs, Stat]. http://arxiv.org/abs/2002.02702
"""

# ╔═╡ 634c9cc1-5a93-42b4-bf51-17dadfe488d6
md"""
## Environment
"""

# ╔═╡ Cell order:
# ╟─5df4d7d2-c622-11eb-3bbd-bff9668ee5e0
# ╟─dceb8312-230f-4e4b-9285-4e23f219b838
# ╟─cda7dc96-d983-4e31-9298-6148205b54b1
# ╟─1436305e-37d8-44f1-88d6-4de838580360
# ╟─08f508c4-233a-4bba-b313-b04c1d6c4a4c
# ╟─868d8932-b108-41d9-b4e8-d62d31b5465d
# ╟─653ec420-8de5-407e-91a9-f045e25a6395
# ╟─716cea7d-d771-46e9-ad81-687292004009
# ╟─cb808fd4-6eb2-457e-afa4-58ae1be09aec
# ╟─0484ae7f-bd8a-4615-a760-5c4b2eef9d3f
# ╟─1d467044-bc7d-4df7-bda6-bb8ea6ff0712
# ╠═b1d99482-53f5-4c6b-8c20-c761ff6bdb77
# ╠═65fa382d-4ef7-432d-8630-27082977185b
# ╠═06f93734-2315-4b36-a39a-09e8167bab1f
# ╟─9f6b96a7-033d-4c7d-a853-46a0b5af4675
# ╟─b7667fb4-6e76-4711-b61d-dae5f993531e
# ╟─cb168dc1-70e2-450f-b2cf-c8680251ab27
# ╟─07d408cf-d202-40b2-90c2-5e8630549339
# ╠═744a8a63-647f-4550-adf7-44354fde44be
# ╟─e6365296-cd68-430e-99c5-fb571f39aad5
# ╠═927ad0a4-ba68-45a6-9bde-561915503e48
# ╟─2ab3c34a-1cfc-4d20-becc-5902d08d03e0
# ╟─924fcad9-75c1-4707-90ef-3e36947d64fe
# ╟─fc8e40c3-34a1-4b2e-bd1b-893d7998d359
# ╠═fb366eb1-4ab0-4e7a-83ed-d531978c06a0
# ╟─5674f7aa-3205-47c7-8367-244c6419ce69
# ╠═83cc80c1-d97e-4b82-872e-e5493d2b62ab
# ╟─c70ebb70-bd96-44a5-85e9-871b0e478b1a
# ╠═dd5fbb2a-4220-4e47-945a-6870b799c50d
# ╟─7d4d06ca-f96d-4b1e-860f-d9e0d6eb6723
# ╠═c64d355f-f5a2-46a5-86f3-2d02da98f305
# ╟─9ebac6ba-d213-4ed8-a1d5-66b841fafa00
# ╠═45c342fd-b893-46aa-b2ee-7c93e7a1d207
# ╟─d44c7baa-80d2-4fdb-a2de-35806477dd58
# ╟─c1b2d007-1004-42f5-b65c-b4e2e7ff7d8e
# ╟─f1153918-0748-4400-ae8b-3b59f8c5d755
# ╟─ad6c4533-cd56-4f6f-b10d-d7bc3145ba16
# ╟─2ef397a6-f7fb-4fc2-b918-40ab545ce19f
# ╟─8d347172-2d26-4d9e-954d-b8924ed4c9e2
# ╟─ca962c0e-4620-4888-b7c3-aa7f6d7899e9
# ╟─6fd49295-d0e3-4b54-aeae-e9cd07a5281c
# ╠═58c5460f-c7f4-4a0a-9e18-71b9580e9148
# ╟─6c04af9b-02af-408d-b9cb-de95ab970f83
# ╟─5d3d2abb-85e3-4371-926e-61ff236253f1
# ╟─247a02e5-8599-43fd-9ee5-32ba8b827477
# ╟─6db0245b-0461-4db0-9462-7a5f80f7d589
# ╠═b5a79826-151e-416e-b0a2-1a58eec9196c
# ╟─9b0b62cb-2c61-4d47-a6c7-09c0c1a75a24
# ╟─9b020402-ea15-4f52-9fff-c70d275b97ac
# ╟─c81f4877-024f-4dc8-b7ce-e781ab6101f3
# ╟─f2272fd5-5132-4a6e-b2ff-136dc2fb2903
# ╟─2d230fea-dcf2-41e6-a477-2a2334f56990
# ╠═44f9935f-c5a5-4f08-a94b-7f6ee70df358
# ╠═39902541-5243-4fa9-896c-36db93d9fcea
# ╟─92e17d42-c6d1-4891-99a9-4a3be9e2decf
# ╠═646ab8dc-db5a-4eb8-a08b-217c2f6d86be
# ╟─5c017766-445d-4f4b-98f1-ae63e78ec34b
# ╠═0a76f019-4853-4ba3-9af8-9f33e1d4c956
# ╟─680f104e-80b4-443f-b4bc-532df758c162
# ╠═ddfc38fc-b47d-4ea5-847a-e9cbee3aa0a1
# ╠═ee2616ca-2602-4823-9cfb-123b958701c4
# ╠═7a62c034-3709-483a-a663-7fe5e09cb773
# ╟─7f1fd9b4-517a-4fec-89bb-4d696dadbc3d
# ╟─81e29fc7-b5d3-46d8-aeac-fb8e6dc11b16
# ╟─5291b260-9a68-4c8b-aff4-7797804ccc95
# ╟─fe0fefb6-2755-4319-a944-bbbc7843aead
# ╟─c109b759-7b73-4593-b9ea-8cc97b61d6fe
# ╟─60494b7c-1a08-4846-8a80-12533552a697
# ╟─b57195f9-c2a1-4676-96f9-faee84f7fc26
# ╟─438d437e-7b00-4a13-8f8a-87fdc332a190
# ╟─26265a91-2c8e-46d8-9a87-a2d097e7433a
# ╠═2eeb402e-c5f9-449c-af19-ff8f2e6c7246
# ╟─2f907e0d-171e-44c3-a531-5f11da08b3cf
# ╠═31b6d4ec-d057-44ca-875b-0c3257895dd3
# ╠═8902a846-fbb9-42fc-8742-c9c4a84db52c
# ╟─b75f8003-85d4-4bb7-96cf-b6d7881b0e7c
# ╠═4af78efd-d484-4241-9d3c-97cc78e1dbd4
# ╟─98ece9fe-dfcc-4dd8-bd47-049217d2afcf
# ╟─634c9cc1-5a93-42b4-bf51-17dadfe488d6
# ╟─31161289-1d4c-46ba-8bd9-e687fb7da29e
