# Grinsztajn, L., Semenova, E., Margossian, C. C., & Riou, J. (2021).
# Bayesian workflow for disease transmission modeling in Stan.
# ArXiv:2006.02985 [q-Bio, Stat].
# http://arxiv.org/abs/2006.02985

using CSV
using DataFrames
using DifferentialEquations
using DiffEqSensitivity
using Distributions
using JLSO
using LazyArrays
using Plots
using Random
using StatsPlots
using Turing
using Dates:now

# Turing.setadbackend(:forwarddiff)
Random.seed!(1)

boarding_school = CSV.read("data/influenza_england_1978_school.csv", DataFrame)

@df boarding_school plot(:date, :in_bed, lw=2, label="I")
@df boarding_school plot!(:date, :convalescent, lw=2, label="R")

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

@model sir(cases, I₀) = begin
  # Calculate number of timepoints
  l = length(cases)
  N = 763
  S₀ = N - I₀
  R₀ = 0

  # Priors
  β ~ TruncatedNormal(2, 1,  1e-6, 100)     # using 100 instead of `Inf` because numerical issues arose
  γ ~ TruncatedNormal(0.4, 0.5,  1e-6, 100) # using 100 instead of `Inf` because numerical issues arose
  ϕ⁻ ~ truncated(Exponential(5), 1, Inf)
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
              # sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
  solᵢ = Array(sol)[2, 2:end] # Infected

  # Likelihood
  for i in 1:l
    solᵢ[i] = max(1e-6, solᵢ[i]) # numerical issues arose
    cases[i] ~ NegativeBinomial(solᵢ[i], ϕ)
  end
end

cases = boarding_school.in_bed
I₀ = first(boarding_school.in_bed)
S₀ = 763 - I₀
R₀ = first(boarding_school.convalescent)

sir_chain = sample(sir(cases, I₀), NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)
summarystats(sir_chain)

JLSO.save("turing/sir_chain.jlso",
          :time => now(),
          :sampler => "NUTS(1_000, 0.65)",
          :specs => "sample(sir(cases, I₀), NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)",
          :seed => 1,
          :chain => sir_chain)
