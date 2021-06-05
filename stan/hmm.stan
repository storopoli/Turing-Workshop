// Leos-Barajas, V., & Michelot, T. (2018).
// An Introduction to Animal Movement Modeling with
// Hidden Markov Models using Stan for Bayesian Inference.
// ArXiv:1806.10639 [q-Bio, Stat].
// http://arxiv.org/abs/1806.10639

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
