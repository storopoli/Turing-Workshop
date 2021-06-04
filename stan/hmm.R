# Example taken from
# https://elevanth.org/blog/2018/01/29/algebra-and-missingness/
library(cmdstanr)
library(here)

set.seed(1)

N_children <- 51
s <- rbinom( N_children , size=1 , prob=0.75 )
s_obs <- s
s_obs[sample(1:N_children, size = 21)] <- -1
tea <- rbinom(N_children, size = 1, prob = s * 1 + (1 - s) * 0.5)

write.csv(data.frame(s, tea), here("data", "tea.csv"), row.names = FALSE)

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
# 1 lp__     -32.7   -32.4   1.16   0.928  -35.0   -31.4    1.00     959.    1298.
# 2 p_cheat    0.566   0.575 0.145  0.150    0.314   0.791  1.00    1430.    1495.
# 3 p_drink    0.950   0.956 0.0331 0.0301   0.886   0.990  1.00     883.    1012.
# 4 sigma      0.792   0.799 0.0639 0.0642   0.683   0.885  1.00    1734.    1365.
