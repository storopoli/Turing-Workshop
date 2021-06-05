# Example taken from
# https://elevanth.org/blog/2018/01/29/algebra-and-missingness/
library(cmdstanr)
library(here)

set.seed(1)

N_children <- 51
s <- rbinom( N_children, size = 1, prob = 0.75 )
s_obs <- s
s_obs[sample(1:N_children, size = 21)] <- -1
tea <- rbinom(N_children, size = 1, prob = s * 1 + (1 - s) * 0.5)

write.csv(data.frame(s_obs, tea), here("data", "tea.csv"), row.names = FALSE)

data_list <- list(
  N_children = N_children,
  tea = tea,
  s = s_obs)

model <- cmdstan_model(here("stan", "hmm.stan"))
fit <- model$sample(data = data_list)
fit$summary()
# A tibble: 4 x 10
#   variable    mean  median     sd    mad      q5     q95  rhat ess_bulk ess_tail
#   <chr>      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl> <dbl>    <dbl>    <dbl>
# 1 lp__     -40.2   -39.9   1.13   0.954  -42.5   -38.9    1.00     944.    1542.
# 2 p_cheat    0.525   0.529 0.135  0.140    0.289   0.736  1.00    1553.    1472.
# 3 p_drink    0.940   0.947 0.0377 0.0368   0.870   0.988  1.00    1110.    1157.
# 4 sigma      0.707   0.711 0.0716 0.0742   0.585   0.816  1.00    1603.    1468.
