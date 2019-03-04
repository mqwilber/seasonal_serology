data {

  int<lower=0> N; // Number of serology samples
  int<lower=0> p; // // Number of degrees of freedom in B-spline basis function not including intercept
  matrix[N, p] X; // Design matrix of B-splines for time of infections/sampling times
  matrix[N, p] Xder; // First derivatives of B-splines
  row_vector[N] infected; // 1 if infected 0 if censored
  vector[N] leftcensored; // 1 if left-censored, 0 otherwise

  // TODO: Could calculate  within Stan so user doesn't have to pass them
  matrix [N, p] Xlc; // B-spline of left-censored hosts

} parameters {

  real<upper=0> intercept; // Intercept should be negative
  positive_ordered[p] betas; // Force ordering to ensure monotonicity of log-cumulative hazard
  real<lower=0> tau; // Regularization parameter

} model {

  vector[N] logcum_hazard;
  vector[N] logsurvival;
  vector[N] loghazard;
  vector[N] logfdensity;
  vector[N] fdensity_trans;

  vector[N] cdf_lc;
  vector[N] cdf_lc_trans;

  // Prior such that the initial value of the survival function should be 
  // close to one with no other information. 

  // The ordered betas are already putting pretty tight constraints.
  betas[1] ~ normal(0, 10);
  for(i in 2:p)
    betas[i] ~ normal(betas[i - 1], tau);
  tau ~ normal(0, 1);


  // Infected calculation
  logcum_hazard = intercept + X*betas; // Linear model on the logcum_hazard scale
  logsurvival = -exp(logcum_hazard);
  loghazard = log(Xder*betas)+ logcum_hazard;
  logfdensity = logsurvival + loghazard;

  cdf_lc = 1 - exp(-exp(intercept + Xlc*betas));

  // Account for left-censoring
  for(i in 1:N){
    cdf_lc_trans[i] = cdf_lc[i]*leftcensored[i];
    fdensity_trans[i] = exp(logfdensity[i])*(1 - leftcensored[i]);
  }

  target += infected*log(fdensity_trans + cdf_lc_trans) + 
            (1 - infected)*logsurvival;
}

