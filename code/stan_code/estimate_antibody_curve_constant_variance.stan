functions{

	real g_delta_theta(real delta, real X1, real A, real r, real B, real d, real X2) {
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

		if(delta < A) {
			ab = X1;
		} else if((delta >= A) && (delta < (A + B))) {

			dA = delta - A;
			ab = X1 + X2*(dA / B) + (r + d*r*(B - dA)) / (B * d^2) -
					 ((r + d*r*B) / (B * d^2)) * exp(-d * dA);
		} else{

			dA = delta - A;
			ab = X1 + X2 + ((r*exp(d*B) - r - d*r*B) / (B*d^2))*exp(-d*dA);
		}

		return ab;

	}

} data {
	int<lower=0> N; // Number of data points in experimental/longitudinal infection data
	real time[N]; // The time since infection in the experimental/longitudinal data
	real antibody[N]; // The antibody data in the experimental/longitudinal data

} parameters {

	// Parameters of the antibody model. Any of these parameter could be moved up 
	// to the data block if they are known or need to be fixed to a certain value
	// when fitting the model.
	real<lower=0> X1;
	real<lower=0> A;
	real<lower=0> r;
	real logB; // B can be large so fit on log scale
	real<lower=0> d;
	real<lower=0> X2; 

	real<lower=0> sigma;

} transformed parameters {
	
	real<lower=0> B;
	B = exp(logB);

} model{

	// Half-normal priors
	sigma ~ normal(0, 1);
	X1 ~ normal(0, 2);
	A ~ normal(0, 2);
	r ~ normal(0, 2);
	logB ~ normal(0, 2); // full normal prior
	d ~ normal(0, 2);
	X2 ~ normal(0, 2);

	// Likelihood of antibody data
	for(i in 1:N){
		target += normal_lpdf(antibody[i] | g_delta_theta(time[i], X1, A, r, exp(logB), d, X2), sigma);
	}
}









