# Stan models for inferring infection risk from serology samples

This folder contains code for inferring seasonal infection from time of infection data.  The estimation of time since infection/time of infection and seasonal infection risk is all done using the probabilistic programming language Stan. Python scripts and Jupyter notebooks are also provided to demonstrate how to move from serology samples to estimating seasonal infection.  Finally, all of the scripts used for the simulation analyses discussed in the main script are provided.

## Stan models for estimating TSI and the antibody curve


- `estimate_TSI_with_longitudinal_and_crosssectional_data.stan`: This Stan model estimates the TSI of seropositive hosts in a cross-sectional sample, using data from longitudinally sampled hosts.  The model jointly estimates the parameters of the antibody curve as well as the TSI for seropositive hosts. **NOTE**: After the fitting the model, the estimates TSIs are held in the parameter `taus`. To obtain the TSI, the following conversion is used: `TSI = tau + max_g`. This is because all TSIs are estimated on the falling arm of the antibody curve and the model is set up such that `taus` are the time since infection after `max_g`, where `max_g` is the time since infection where the antibody quantity is maximized.

- `estimate_TSI_with_longitudinal_data.stan`: This Stan model estimates the TSI of seropositive hosts in a cross-sectional sample. In contrast to `estimate_TSI_with_longitudinal_and_crosssectional_data.stan`, this model assumes that the parameters of the antibody curve are already known or estimated and takes these parameters in as data.  **NOTE**: After the fitting the model, the estimates TSIs are held in the parameter `taus`. To obtain the TSI, the following conversion is used: `TSI = tau + max_g`. This is because all TSIs are estimated on the falling arm of the antibody curve and the model is set up such that `taus` are the time since infection after `max_g`, where `max_g` is the time since infection where the antibody quantity is maximized.

- `estimate_antibody_curve_changing_variance.stan`: This Stan model estimates the parameters of the antibody curve given longitudinal antibody data. The model assumes that variance around the antibody quantity is changing with time since infection.

- `estimate_antibody_curve_constant_variance.stan`: This Stan model estimates the parameters of the antibody curve given longitudinal antibody data. The model assumes constant variance around the antibody quantity.


## Stan models for inferring infection risk from TSI estimates using survival analysis

- `infectionrisk_rightcensored.stan`: Equation 1 in the main text. The standard right-censored survival analysis for time of infection data with a flexible, parametric function for seasonal infection risk.

- `infectionrisk_age.stan`: Equation 2 in the main text. A right-censored, survival analysis that accounts for host age (i.e. date of birth) when estimating seasonal infection risk.infection 

- `infectionrisk_leftcensored.stan`: Equation 3 in the main text. A survival analysis that accounts for both left-censored infected hosts (i.e. hosts with an elevated antibody level), hosts with a time of infection, and right-censored uninfected hosts.
 
- `infectionrisk_leftcensored_age.stan`: Equation 4 in the main text.  A survival analysis that accounts for right-censoring, left-censoring, host age, and time of infections. This is the likelihood that was used when analyzing the feral swine data.

- `infectionrisk_leftcensored_rf.stan`: Appedix S4: Equation S4. A survival analysis that simultaneously accounts for right-censoring, left-censoring, and the rise and fall of the antibody curve on time of infection. 

- `infectionrisk_recovery.stan`: Appendix S6: Equation S1. A survival analysis that accounts for right-censoring and apparent recovery due to a seroconversion threshold. 

