data{
  int N_children;      // number of children
  int tea[N_children]; // [0,1] observed drinking tea
  int s[N_children];   // [0,1,-1] stabled ox
}
parameters{
  real p_cheat;
  real p_drink;
  real sigma;
}
model{
  // priors
  p_cheat ~ beta(2,2);
  p_drink ~ beta(2,2);
  sigma ~ beta(2,2);

  // probability of tea
  for ( i in 1:N_children ) {
    if ( s[i] == -1 ) {
      // ox unobserved
      target += log_mix(
                  sigma ,
                  bernoulli_lpmf( tea[i] | p_drink ) ,
                  bernoulli_lpmf( tea[i] | p_cheat ) );
      // could also be
      // target += log_sum_exp(
      //             log(sigma) + bernoulli_lpmf( tea[i] | p_drink ) ,
      //             log(1 - sigma) + bernoulli_lpmf( tea[i] | p_cheat ) );

    } else {
      // ox observed
      tea[i] ~ bernoulli( s[i]*p_drink + (1-s[i])*p_cheat );
      s[i] ~ bernoulli( sigma );
    }
  }
}
