data {

  int<lower=0> N; // Number of serology samples
  int<lower=0> p; // Number of degrees of freedom in B-spline basis function not including intercept
  matrix[N, p] X; // Design matrix of B-splines
  matrix[N, p] Xder; // First derivatives of B-splines

  matrix [N, p] Xdob; // Left-truncation vector based on date of birth
  row_vector[N] infected; // 1 if infected 0 if censored

} parameters {

  real<upper=0> intercept; // Remove upper bound if necessary
  positive_ordered[p] betas; // Force ordering to ensure monotonicity of log-cumulative hazard
  real<lower=0> tau; // Regularization parameter

}  model {

  vector[N] logcum_hazard;
  vector[N] logsurvival;
  vector[N] loghazard;
  vector[N] logfdensity;

  vector[N] cdf_dob;

  // Prior such that the initial value of the survival function should be 
  // close to one with no other information.  Assuming that infection
  // risk starts close to zero.
  intercept ~ normal(-5, 1);

  betas[1] ~ normal(0, 10);
  for(i in 2:p)
    betas[i] ~ normal(betas[i - 1], tau);
  tau ~ normal(0, 1);

  logcum_hazard = intercept + X*betas; // Linear model on the logcum_hazard scale
  logsurvival = -exp(logcum_hazard);
  loghazard = log(Xder*betas)+ logcum_hazard;
  logfdensity = logsurvival + loghazard;

  cdf_dob = 1 - exp(-exp(intercept + Xdob*betas));

  // Left-truncated likelihood based on date of birth
  target += infected*(logfdensity) + 
            (1 - infected)*log((exp(logsurvival) + cdf_dob));
}

