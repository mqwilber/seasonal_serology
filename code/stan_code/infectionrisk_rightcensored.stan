// Flexible parametric right-censored survival analysis.  Equation 1 from the 
// main text. 
data {

  int<lower=0> N; // Number of serology samples
  int<lower=0> p; // Number of degrees of freedom in B-spline basis function not including intercept
  matrix[N, p] X; // Design matrix of B-splines
  matrix[N, p] Xder; // First derivatives of B-splines
  row_vector[N] infected; // 1 if infected 0 if censored

} parameters {

  real<upper=0> intercept; // Intercept should be negative
  positive_ordered[p] betas; // Force ordering to ensure monotonicity of log-cumulative hazard
  real<lower=0> tau; // Regularization parameter

}  model {

  vector[N] logcum_hazard;
  vector[N] logsurvival;
  vector[N] loghazard;
  vector[N] logfdensity;

  betas[1] ~ normal(0, 10);
  for(i in 2:p)
    betas[i] ~ normal(betas[i - 1], tau);
  tau ~ normal(0, 1);

  logcum_hazard = intercept + X*betas; // Linear model on the logcum_hazard scale
  logsurvival = -exp(logcum_hazard);
  loghazard = log(Xder*betas)+ logcum_hazard;
  logfdensity = logsurvival + loghazard;

  // Log-likelihood data
  target += infected*logfdensity + (1 - infected)*logsurvival;
}

