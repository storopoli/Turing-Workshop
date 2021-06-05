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
    K = 2)

write.csv(data.frame(y, S), here("data", "hmm.csv"), row.names = FALSE)

model <- cmdstan_model(here("stan", "hmm.stan"))
fit <- model$sample(data = data_list)
fit$summary()

# A tibble: 7 x 10
#   variable         mean     median     sd    mad         q5       q95  rhat ess_bulk ess_tail
#   <chr>           <dbl>      <dbl>  <dbl>  <dbl>      <dbl>     <dbl> <dbl>    <dbl>    <dbl>
# 1 lp__       -2348.     -2348.     1.40   1.28   -2351.     -2346.     1.00    1675.    1953.
# 2 mu[1]          1.18       1.18   0.112  0.113      0.986      1.36   1.00    2536.    2722.
# 3 mu[2]          5.07       5.07   0.108  0.108      4.89       5.25   1.00    4808.    3180.
# 4 theta[1,1]     0.900      0.901  0.0176 0.0173     0.869      0.927  1.00    3154.    2785.
# 5 theta[2,1]     0.0995     0.0987 0.0170 0.0173     0.0729     0.128  1.00    3384.    2628.
# 6 theta[1,2]     0.100      0.0991 0.0176 0.0173     0.0727     0.131  1.00    3155.    2785.
# 7 theta[2,2]     0.901      0.901  0.0170 0.0173     0.872      0.927  1.00    3384.    2628.
