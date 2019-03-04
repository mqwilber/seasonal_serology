functions{

	real g_tau_theta(real tau, real X1, real A, real r, real B, real d, real X2) {
		// The antibody function g(\tau, \theta) as given by equation S1 in the 
		// the supplementary material.
		//
		// Parameters
		// ----------
		//  tau : time since infection. 
		// theta => X1, A, r, B, d, X2
		// 	X1: The baseline level of antibodies prior to exposure, 
		//  A: The initial lag between exposure and antibody production
		//  r: The antibody production rate
		//  d: The antibody decay rate
		//  B: The period of antibody production in response to infection
		//  X2: X_1 + X_2 gives the baseline antibody level following antibody decay.
		//
		// Returns
		// -------
		// ab: the antibody level tau (tau) time units after exposure given theta
		real ab;
		real dA;

		if(tau < A) {
			ab = X1;
		} else if((tau >= A) && (tau < (A + B))) {

			dA = tau - A;
			ab = X1 + X2*(dA / B) + (r + d*r*(B - dA)) / (B * d^2) -
					 ((r + d*r*B) / (B * d^2)) * exp(-d * dA);
		} else{

			dA = tau - A;
			ab = X1 + X2 + ((r*exp(d*B) - r - d*r*B) / (B*d^2))*exp(-d*dA);
		}

		return ab;

	}

} data {
	int<lower=0> N; // Number of data points in experimental/longitudinal infection data
	real time[N]; // The time since infection in the experimental/longitudinal data
	real antibody[N]; // The antibody data in the experimental/longitudinal data

} transformed data {

	// Transform time to help with estimation
	real mean_time; 
	real sd_time;
	real trans_time[N];

	mean_time = mean(time);
	sd_time = sd(time);

	for(i in 1:N){
		trans_time[i] = (time[i] - mean_time) / sd_time;
	}

} parameters {

	// Parameters of the antibody model. Any of these parameter could be moved up 
	// to the data block if they are known or need to be fixed to a certain value
	// when fitting the model.
	real<lower=0> X1;
	real<lower=0> A;
	real<lower=0> r;
	real<lower=0> d;
	real<lower=0> X2;
	real<lower=0> B;


	real<lower=0> sigma;
	real beta0; // Parameter allowing variance to scale
	real beta1; // Parameter allowing variance to scale

} model{

	real sigma_full;

	// Priors
	sigma ~ cauchy(0, 1);
	beta0 ~ normal(0, 2);
	beta1 ~ normal(0, 2);
	X1 ~ cauchy(0, 1);
	A ~ cauchy(0, 1);
	r ~ cauchy(0, 1);
	d ~ cauchy(0, 1);
	X2 ~ cauchy(0, 1);
	B ~ cauchy(0, 1);

	// Likelihood of antibody data
	for(i in 1:N){
		sigma_full = sigma^2 * (1 - exp(-exp(beta0 + beta1*trans_time[i])));
		target += normal_lpdf(antibody[i] | g_tau_theta(time[i], X1, A, r, B, d, X2), sqrt(sigma_full));
	}
}