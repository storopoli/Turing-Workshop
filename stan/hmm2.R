# Leos-Barajas, V., & Michelot, T. (2018).
# An Introduction to Animal Movement Modeling with
# Hidden Markov Models using Stan for Bayesian Inference.
# ArXiv:1806.10639 [q-Bio, Stat].
# http://arxiv.org/abs/1806.10639
library(cmdstanr)

set.seed(1)

# Number of states
N <- 2
# transition probabilities
Gamma <- matrix(c(0.9, 0.1, 0.1, 0.9), 2, 2)
# initial distribution set to the stationary distribution
delta <- solve(t(diag(N)-Gamma +1), rep(1, N))
# state-dependent Gaussian means
mu <- c(1, 5)

nobs <- 1000
S <- rep(NA, nobs)
y <- rep(NA, nobs)

# initialise state and observation
S[1] <- sample(1:N, size = 1, prob = delta)
y[1] <- rnorm(1, mu[S[1]], 2)

# simulate state and observation processes forward
for (t in 2:nobs) {
    S[t] <- sample(1:N, size = 1, prob = Gamma[S[t - 1], ])
    y[t] <- rnorm(1, mu[S[t]], 2)
}

data_list <- list(
    y = y,
    T = nobs,
    N = 2)

write.csv(data.frame(y), here("data", "hmm.csv"), row.names = FALSE)

model <- cmdstan_model(here("stan", "hmm2.stan"))
fit <- model$sample(data = data_list)
fit$summary()

# A tibble: 13 x 10
#    variable         mean    median     sd    mad         q5       q95  rhat ess_bulk ess_tail
#    <chr>           <dbl>     <dbl>  <dbl>  <dbl>      <dbl>     <dbl> <dbl>    <dbl>    <dbl>
#  1 lp__        -2351.    -2351.    1.42   1.20   -2354.     -2350.     1.00    2250.    2932.
#  2 theta[1,1]      0.899     0.900 0.0178 0.0179     0.868      0.926  1.00    2656.    2576.
#  3 theta[2,1]      0.101     0.101 0.0171 0.0170     0.0750     0.132  1.00    2956.    2914.
#  4 theta[1,2]      0.101     0.100 0.0178 0.0179     0.0745     0.132  1.00    2656.    2576.
#  5 theta[2,2]      0.899     0.899 0.0171 0.0170     0.868      0.925  1.00    2956.    2914.
#  6 mu[1]           1.17      1.17  0.109  0.112      0.990      1.35   1.00    2868.    2682.
#  7 mu[2]           5.08      5.08  0.110  0.112      4.90       5.26   1.00    4867.    3040.
#  8 ta[1,1]         0.899     0.900 0.0178 0.0179     0.868      0.926  1.00    2656.    2576.
#  9 ta[2,1]         0.101     0.101 0.0171 0.0170     0.0750     0.132  1.00    2956.    2914.
# 10 ta[1,2]         0.101     0.100 0.0178 0.0179     0.0745     0.132  1.00    2656.    2576.
# 11 ta[2,2]         0.899     0.899 0.0171 0.0170     0.868      0.925  1.00    2956.    2914.
# 12 statdist[1]     0.501     0.500 0.0488 0.0490     0.421      0.582  1.00    3623.    3022.
# 13 statdist[2]     0.499     0.500 0.0488 0.0490     0.418      0.579  1.00    3623.    3022.
