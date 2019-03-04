// This Stan model estimates the TSI of seropositive hosts in a cross-sectional
// sample. In contrast to
// `estimate_TSI_with_longitudinal_and_crosssectional_data.stan`, this model
// assumes that the parameters of the antibody curve are already known or
// estimated and takes these parameters in as data.  **NOTE**: After the fitting
// the model, the estimates TSIs are held in the parameter `taus`. To obtain the
// TSI, the following conversion is used: `TSI = tau + max_g`. This is because all
// TSIs are estimated on the falling arm of the antibody curve and the model is
// set up such that `taus` are the time since infection after `max_g`, where
// `max_g` is the time since infection where the antibody quantity is maximized.

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


	real max_g(real A, real r, real B, real d, real X2){
		// Computes the time at which g is maximized. Parameter are as defined above

		real maxt;
		maxt = (1 / -d)*(log((r / (B*d)) - X2 / B) - (log((r + d*r*B) / (B*d)) + d*A));
		return maxt;

	}

} data {
	int<lower=0> N; // Number of data points in experimental/longitudinal infection data
	real time[N]; // The time since infection in the experimental/longitudinal data
	real antibody[N]; // The antibody data in the experimental/longitudinal data

	int<lower=0> P; // Number of positive samples from the cross-sectional data
	int<lower=0> T; // Number of data points from cross-sectional samples.  All are assumed to be unique individuals
	// int<lower=0> S; // Number of regions

	real ab[T]; // Antibody samples
	int positive_id[T]; // All seropositive individuals are labeled 1 - P and seronegative individuals are labeled -1
	int positive[T]; // 1 if infected and 0 if uninfected

	// Parameters for the antibody curve. In this case, they are assumed to be known
	// or previously estimated and are not estimated jointly with TSI from the
	// longitudinal daata 
	real<lower=0> X2;
	real<lower=0> B;
	real<lower=0> X1;
	real<lower=0> A;
	real<lower=0> r;
	real<lower=0> d;
	real<lower=0> sigma;

} parameters {

	// TSI parameters
	real<lower=0> taus[P]; // Time since infections for cross section data
	real<lower=0> lambdas; // Regularization effect 

} model{

	real maxt;

	// This is a hierarchical model on TSI.
	lambdas ~ normal(0, 1);
 
	// Loop through field data
	for(j in 1:T){


		if(positive[j] == 0){


			// UNCOMMENT TO ALLOW THE NEGATIVES TO CONTRIBUTE TO THE ESTIMATE OF SIGMA	
			// target += normal_lpdf(ab[j] | g_tau_theta(0, X1, A, r, B, d, X2), sigma);

		} else{

			// Regularization of taus
			target += exponential_lpdf(taus[positive_id[j]] | lambdas);

			maxt = max_g(A, r, B, d, X2);
			target += normal_lpdf(ab[j] | g_tau_theta(taus[positive_id[j]] + maxt, X1, A, r, B, d, X2), sigma);

		}
	}
}









