// Leos-Barajas, V., & Michelot, T. (2018).
// An Introduction to Animal Movement Modeling with
// Hidden Markov Models using Stan for Bayesian Inference.
// ArXiv:1806.10639 [q-Bio, Stat].
// http://arxiv.org/abs/1806.10639

data {
  int<lower=0> N; // number of states
  int<lower=1> T; // length of data set
  real y[T]; // observations
}
parameters {
  simplex[N] theta[N]; // N x N tpm
  ordered[N] mu; // state-dependent parameters
}
transformed parameters{
  matrix[N, N] ta; //
  simplex[N] statdist; // stationary distribution
  for(j in 1:N){
    for(i in 1:N){
      ta[i,j]= theta[i,j];
    }
}
  statdist =  to_vector((to_row_vector(rep_vector(1.0, N))/
      (diag_matrix(rep_vector(1.0, N)) - ta + rep_matrix(1, N, N)))) ;
}
model {
  vector[N] log_theta_tr[N];
  vector[N] lp;
  vector[N] lp_p1;
  // prior for mu
  mu ~ student_t(3, 0, 1);
  // transpose the tpm and take natural log of entries
  for (n_from in 1:N)
  for (n in 1:N)
    log_theta_tr[n, n_from] = log(theta[n_from, n]);
  // forward algorithm implementation
  for(n in 1:N) // first observation
    lp[n] = log(statdist[n]) + normal_lpdf(y[1] | mu[n], 2);
  for (t in 2:T) { // looping over observations
    for (n in 1:N) // looping over states
      lp_p1[n] = log_sum_exp(log_theta_tr[n] + lp) +
        normal_lpdf(y[t] | mu[n], 2);
lp = lp_p1; }
  target += log_sum_exp(lp);
}
