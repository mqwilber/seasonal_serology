data {

  int<lower=0> N; // // Number of serology samples
  int<lower=0> p; // Number of degrees of freedom in B-spline basis function not including intercept
  matrix[N, p] X; // Design matrix of B-splines for time of infections/sampling times
  matrix[N, p] Xder; // First derivatives of B-splines
  row_vector[N] infected; // 1 if infected 0 if censored
  vector[N] recovery; // 1 if potentially recovered, 0 otherwise

  // TODO: calculate these within Stan so user doesn't have to pass them
  // Currently this are calculated externally in build_parametric_data in
  // survival.py
  matrix [N, p] Xnu; // B-spline space for nu, which is t - tau_falling
  matrix [N, p] Xlower; // B-spline space for lower, which is t - tau_rising

} parameters {

  real<upper=0> intercept; // Remove upper bound if necessary
  positive_ordered[p] betas; // Force ordering to ensure monotonicity of log-cumulative hazard
  real<lower=0> tau; // Regularization parameter

} model {

  vector[N] logcum_hazard;
  vector[N] logsurvival;
  vector[N] loghazard;
  vector[N] logfdensity;

  vector[N] cdf_nu;
  vector[N] cdf_nu_trans;
  vector[N] survival_lower;

  // Let the intercept be unconstrained. You can set a prior on the intercept
  // to force it to be more negative, ensuring that infection risk starts at
  // 0. e.g.
  // intercept ~ normal(-5, 1);

  // Regularlization of betas
  betas[1] ~ normal(0, 10);
  for(i in 2:p)
    betas[i] ~ normal(betas[i - 1], tau);
  tau ~ normal(0, 1);

  // Infected calculation
  logcum_hazard = intercept + X*betas; // Linear model on the logcum_hazard scale
  logsurvival = -exp(logcum_hazard);
  loghazard = log(Xder*betas)+ logcum_hazard;
  logfdensity = logsurvival + loghazard;

  cdf_nu = 1 - exp(-exp(intercept + Xnu*betas));
  survival_lower = exp(-exp(intercept + Xlower*betas));

  // Only use the CDF if recovery could have occurred
  for(i in 1:N)
    cdf_nu_trans[i] = cdf_nu[i]*recovery[i];

  target += infected*logfdensity + (1 - infected)*log(survival_lower + cdf_nu_trans);
}

