## Python 3.6.8
import numpy as np # Version '1.14.2'
import survival as surv
import scipy.stats as stats # Version '1.0.0'
import matplotlib.pyplot as plt # Version '2.2.2'
import seaborn as sns # Version '0.8.1'
from patsy import dmatrix, bs, build_design_matrices # Version '0.5.1'
from patsy.splines import BS  # Version '0.5.1'
import pandas as pd # Version '0.23.4'
import pystan # Version '2.18.1.0'

colors = sns.color_palette() 

"""
Description
-----------

Evaluating how accounting for host age can improve estimates of infection risk.
This script simulated uniform host birth over the course of the sampling season
as well as a unimodal infection process and then estimates seasonal infection
risk both accounting for and not accounting for host age.

Author
------
Mark Q. Wilber


"""

def simulate_data_with_births(samp_size, wei_params, start, percent_adult, max_days):
  """
  Simulate pig infection data based on a Weibull distribution and a uniform
  DOB distribution with sum percent of pigs being born before the the 
  interval of interest (adults)

  Parameters
  ----------
  samp_size : int
    Total number of hosts to sample
  wei_params : tuple
    alpha, sigma parameters of Weibull distribution
  start : pd.DateTime
    The start date of the infection season
  percent_adult : float
    Percent of pigs that are adults i.e. could be infected before the start of sampling.
  max_days : int
    The number of days from the start of the infection season at which the season
    ends

  Returns
  -------
  : DataFrame 
    Infection times, infection indicator, sample_date, dob

  """

  wei = stats.weibull_min(wei_params[0], scale=wei_params[1])

  adults = np.int(samp_size * percent_adult)
  dobs = np.append(np.repeat(0, adults), 
                  stats.uniform(loc=0, scale=max_days).rvs(size=samp_size - adults))

  upper_prob = wei.cdf(max_days)
  lower_probs = wei.cdf(dobs)
  inf_probs = upper_prob - lower_probs
  inf = stats.binom.rvs(1, inf_probs, size=len(inf_probs))

  # For each infected individual draw a toi
  unifrand = stats.uniform.rvs(loc=lower_probs, scale=upper_prob - lower_probs, 
                               size=len(inf_probs))
  toi_samp = wei.ppf(unifrand)

  sampling_time = stats.uniform(loc=dobs, scale=max_days - dobs).rvs(size=samp_size)
  eventobserved = ((sampling_time > toi_samp) & (inf == 1)).astype(np.int)
  time = np.where(eventobserved == 1, toi_samp, sampling_time)

  surv_df = pd.DataFrame(dict(time_in_season=time, infection=eventobserved == 1, 
                 sample_date = start + pd.to_timedelta(sampling_time, unit="D"),
                 dob=start + pd.to_timedelta(dobs, unit="D")))

  return(surv_df)


if __name__ == '__main__':
  

  samp_size = 1000
  percent_adult = 0.8 # % of the population that is present before samplings
  alpha = 4 # Weibull parameter
  sigma = 150 # Weibull parameter
  start = pd.Timestamp(pd.datetime(2016, 1, 1))
  max_days = 365
  df = 4
  theta = dict(X1=0.06, A=3.39, r=0.05, d=0.08, B=150, X2=0.07)

  surv_df = simulate_data_with_births(samp_size, (alpha, sigma), start, 
                                      percent_adult, max_days)

  # No recovery or left-censoring in this simulation
  recovered = np.repeat(0, len(surv_df))
  leftcensored = recovered
  standata = surv.build_parametric_data(surv_df, recovered, leftcensored, 
                                        theta, start, df, calc_dob=True)

  # Build pystan models
  print("Building stan models...")
  base = pystan.StanModel("stan_code/infectionrisk_age.stan")
  nodob = pystan.StanModel("stan_code/infectionrisk_rightcensored.stan")

  # Fit pystan models
  basefit = base.sampling(standata, iter=4000, warmup=2000, chains=2)
  nodobfit = nodob.sampling(standata, iter=4000, warmup=2000, chains=2)

  # Plot results
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))
  axes = axes.ravel()

  # Plot the truth 
  vals = np.linspace(0, 365, num=100)
  wei = stats.weibull_min(alpha, scale=sigma)
  axes[1].plot(vals, wei.cdf(vals), color="black")
  axes[0].plot(vals, wei.pdf(vals), color="black", label="Truth")

  time = surv_df.time_in_season.values

  base_curves = surv.extract_fit(basefit, time, df)
  nodob_curves =  surv.extract_fit(nodobfit, time, df)

  curves = [base_curves, nodob_curves]
  labels = ['Accounting for age', "Not accounting for age"]
  cs = [colors[2], colors[1]]
  style = ['-', '--']

  for i in range(len(curves)):

    c = curves[i]
    med_f = np.exp(stats.scoreatpercentile(c['logfdensity'], 50, axis=0)) / np.max(time)
    lower_f = np.exp(stats.scoreatpercentile(c['logfdensity'], 2.5, axis=0)) / np.max(time) 
    upper_f = np.exp(stats.scoreatpercentile(c['logfdensity'], 97.5, axis=0)) / np.max(time)

    med_s = np.exp(stats.scoreatpercentile(c['logsurvival'], 50, axis=0))
    lower_s = np.exp(stats.scoreatpercentile(c['logsurvival'], 2.5, axis=0))
    upper_s = np.exp(stats.scoreatpercentile(c['logsurvival'], 97.5, axis=0))

    axes[0].plot(c['newtime'], med_f, label=labels[i], ls=style[i], color=cs[i])
    axes[0].fill_between(c['newtime'], lower_f, upper_f, alpha=0.2, color=cs[i])

    axes[1].plot(c['newtime'], 1 - med_s, ls=style[i], color=cs[i])
    axes[1].fill_between(c['newtime'], 1 - upper_s, 1 - lower_s, alpha=0.2, 
                          color=cs[i])

  axes[0].legend(loc="upper right", prop={'size': 7})
  axes[0].set_xlabel("Days since start of infection")
  axes[1].set_xlabel("Days since start of infection")
  axes[1].set_ylabel("Probability of infection by t, F(t)")
  axes[0].set_ylabel("Incidence, f(t)")

  fig.savefig("../results/accounting_for_age.pdf", bbox_inches="tight")






