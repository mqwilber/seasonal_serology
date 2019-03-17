data {

  int<lower=0> N; // // Number of serology samples
  int<lower=0> p; // Number of degrees of freedom in B-spline basis function not including intercept
  matrix[N, p] Xrise; // Design matrix of B-splines for rising
  matrix[N, p] Xder_rise; // First derivatives of B-splines for rising
  matrix[N, p] X; // Design matrix of B-splines for falling values
  matrix[N, p] Xder; // First derivatives of B-splines for falling values
  vector[N] pfalling; // Falling probability
  vector[N] prising; // Rising probability
  row_vector[N] infected; // 1 if infected 0 if censored
  vector[N] leftcensored; // 1 if left censored, 0 otherwise

  // TODO: calculate these within Stan so user doesn't have to pass them
  matrix [N, p] Xlc; // B-spline of left-censored hosts
  matrix [N, p] Xlower; // B-spline space for lower. If tau_rising = 0, Xlower is Xder and equation S7 in the appendix is recovered

} parameters {

  real<upper=0> intercept; // Remove upper bound if necessary
  positive_ordered[p] betas; // Force ordering to ensure monotonicity of log-cumulative hazard
  real<lower=0> tau;

} model {

  vector[N] logcum_hazard_rise;
  vector[N] logsurvival_rise;
  vector[N] loghazard_rise;
  vector[N] logfdensity_rise;

  vector[N] logcum_hazard_fall;
  vector[N] logsurvival_fall;
  vector[N] loghazard_fall;
  vector[N] logfdensity_fall;

  vector[N] cdf_lc;
  vector[N] cdf_lc_trans;

  vector[N] fdensity_trans;
  vector[N] survival_lower;

  // Let the intercept be unconstrained
  betas ~ normal(0, 10);
  for(i in 2:p)
    betas[i] ~ normal(betas[i - 1], tau); 
  tau ~ normal(0, 1);

  // Compute rising probabilities
  logcum_hazard_rise = intercept + Xrise*betas; // Linear model on the logcum_hazard scale
  logsurvival_rise = -exp(logcum_hazard_rise);
  loghazard_rise = log(Xder_rise*betas)+ logcum_hazard_rise;
  logfdensity_rise = logsurvival_rise + loghazard_rise;

  // Compute falling probabilities
  logcum_hazard_fall = intercept + X*betas; // Linear model on the logcum_hazard scale
  logsurvival_fall = -exp(logcum_hazard_fall);
  loghazard_fall = log(Xder*betas)+ logcum_hazard_fall;
  logfdensity_fall = logsurvival_fall + loghazard_fall;

  cdf_lc = 1 - exp(-exp(intercept + Xlc*betas));
  survival_lower = exp(-exp(intercept + Xlower*betas));

  // Account for left_censoring and rise and fall probabilities
  for(i in 1:N){
    cdf_lc_trans[i] = cdf_lc[i]*leftcensored[i];
    fdensity_trans[i] = (prising[i]*exp(logfdensity_rise[i]) + pfalling[i]*exp(logfdensity_fall[i]))*(1 - leftcensored[i]);
  }

  target += infected*log(fdensity_trans + cdf_lc_trans) + (1 - infected)*log(survival_lower);
}

