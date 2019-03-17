## Python 3.6.8
import numpy as np # Version '1.14.2'
import survival as surv 
import pandas as pd # Version '0.23.4'
import scipy.stats as stats # Version '1.0.0'
import matplotlib.pyplot as plt # Version '2.2.2'
import seaborn as sns # Version '0.8.1'
import pystan  # Version '2.18.1.0'
import os

"""
Description
-----------

Exploring the ability of a modified survival analysis to account for 
left-censoring in some or all of seropositive hosts. "Left-censored" means a host 
is seropositive but doesn't have an estimated time of infection. 

Author
------
Mark Q. Wilber

"""

if __name__ == '__main__':

  # Some date preliminary parameters
  start_month = 5
  end_month = 4
  start = pd.datetime(2015, start_month, 1)
  end = pd.datetime(2016, end_month, 30)
  inf_days = pd.date_range(start, end)
  vals = np.arange(0, 365)

  # Parameters for the antibody curve and the cutoff
  theta = dict(X1=0.06, A=3.39, r=0.05, d=0.08, B=80, X2=0.14)
  sigma = 0.00
  tf = lambda x: np.log10(1 / x)
  cutoff = tf(0.85)#tf(0.673)
  tau_rising, tau_falling = surv.inverse_antibody_dynamics(cutoff, theta)
  tau_end = surv.get_tau_end(theta)

  # Number of pigs in the entire region/population
  num_pop = 10000 

  # Number of pigs to sample from the full distribution
  num_pigs_array = [1000]

  # Set up different time of infection probability distributions
  days = np.array([d.days for d in pd.to_datetime(inf_days) - start])
  unif = np.repeat(1 / (len(inf_days)), len(inf_days))
  unimod = surv.weibull_sampling_probs(days, 3, 200)
  bimod = surv.mixture_weibull_probs(days, [5, 10], [100, 250], [1, 1])
  all_probs = [unif, unimod, bimod, unif, unimod, bimod, unif, unimod, bimod]
  names = ["Uniform", "Unimodal", "Bimodal"]
  dfs = [4, 4, 12]

  # Build stan model
  print("Building stan models...")
  nm = "stan_code/infectionrisk_leftcensored.pkl"
  if os.path.exists(nm):
    weibull_adj = pd.read_pickle(nm)
  else:
    weibull_adj = pystan.StanModel("stan_code/infectionrisk_leftcensored.stan")
    pd.to_pickle(weibull_adj, nm)
  print("Completed building stan models")

  # Loop over all combinations of distributions

  for num_pigs in num_pigs_array:

    print("Working on {0} pigs sampled...".format(num_pigs))

    all_truth = []
    all_trun = []
    all_wadj = []

    for i in range(3): # TOI distributions

      # Build serology data set
      sero_df = surv.build_serology_data(inf_days, all_probs[i], all_probs[0], 
                               num_pigs, theta, sigma, start,
                               total_pop=num_pop)
      sero_df = sero_df.assign(seropositive=lambda x: x.serodata > cutoff)

      # Assuming no uncertainty in time of infection
      sind = sero_df.seropositive.values
      sero_df['toi'] = pd.NaT
      sero_df.loc[sind, 'toi'] = sero_df.true_toi[sind]
      sero_df.loc[~sind, 'toi'] = sero_df.sample_date[~sind] + pd.Timedelta(1, unit="D")

      # Identify left-censored observations
      left_censored = np.isclose(sero_df.serodata.values, theta['X1'] + theta['X2'], rtol=0.01)
      sample_times = np.array([d.days for d in sero_df.sample_date - start])
      left_censored = (pd.Series(left_censored) & (pd.Series(sample_times) > tau_end)).values
      sero_df.loc[left_censored, "toi"] = sero_df.sample_date[left_censored] - pd.Timedelta(tau_end, unit="D")
      recovered = np.repeat(0, len(left_censored))

      est_df = sero_df[['sample_date', 'toi', "serodata"]]
      est_df_surv = surv.build_survival_data(est_df, start)

      df = dfs[i]
      standata = surv.build_parametric_data(est_df_surv, 
                                             recovered, 
                                             left_censored.astype(np.int),
                                             theta,
                                             start,
                                             df)

      wfit_tau = weibull_adj.sampling(data=standata, iter=4000, warmup=2000, chains=2)
      wcurves = surv.extract_fit(wfit_tau, est_df_surv.time_in_season.values, df)
      all_wadj.append(wcurves)

      # Get the truth with out truncation
      true_tsi = np.array([d.days for d in  sero_df.true_toi - start])
      true_ecdf = surv.empirical_cdf(true_tsi)
      all_truth.append(true_ecdf)

      # Get the truth with truncation
      trun_tsi = true_tsi[~left_censored]
      trun_ecdf = surv.empirical_cdf(trun_tsi)
      all_trun.append(trun_ecdf)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes = axes.ravel()

    count = 0
    for i, ax in enumerate(axes):

      true_ecdf = all_truth[i]
      trun_ecdf = all_trun[i]
      fitvals = all_wadj[i]

      # Extract the Weibull estimates
      median_wei = 1 - np.exp(stats.scoreatpercentile(fitvals['logsurvival'], 50, axis=0))
      lower_wei = 1 - np.exp(stats.scoreatpercentile(fitvals['logsurvival'], 2.5, axis=0))
      upper_wei = 1 - np.exp(stats.scoreatpercentile(fitvals['logsurvival'], 97.5, axis=0))
      newtime = fitvals['newtime']

      ax.plot(true_ecdf.data, true_ecdf.ecdf, lw=2, color="black", label="Truth")
      ax.plot(trun_ecdf.data, trun_ecdf.ecdf, lw=2, color=sns.color_palette()[1], ls="dashed", label="Excluding left-censored")

      ax.plot(newtime, median_wei, label="Accounting for\nleft-censored",
                                        color=sns.color_palette()[2])

      ax.fill_between(newtime, upper_wei, lower_wei, 
                                color=sns.color_palette()[2], alpha=0.2)

      ax.set_xticks(np.arange(0, 365, 75))

      if i == 1:
        ax.set_xlabel("Days since start of infection season", size=15)

      if i == 0:
        ax.set_ylabel("Probability of infection by t, F(t)", size=15)

      if ax.is_first_row():
        ax.set_title(names[i], size=12)

      if i == 1:
        ax.legend(prop={'size': 6.5}, frameon=False)

      plt.show()
      fig.savefig("../results/test_leftcensored_n{0}.pdf".format(num_pigs), bbox_inches="tight")



