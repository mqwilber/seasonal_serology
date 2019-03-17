## Python 3.6.8
import numpy as np # Version '1.14.2'
import survival as surv
import pandas as pd # Version '0.23.4'
import scipy.stats as stats # Version '1.0.0'
import matplotlib.pyplot as plt  # Version '2.2.2'
import seaborn as sns  # Version '0.8.1'
import pystan # Version '2.18.1.0'
import importlib
from patsy import dmatrix, build_design_matrices # Version '0.5.1'
from patsy.splines import BS # Version '0.5.1'
import os

"""
Description
-----------

Exploring the ability of survival analysis to recover the time of infection 
distributions under different sampling distribution efforts.

Author
------
Mark Q. Wilber

"""

def build_standata(ss_df, df, degree=3, include_intercept=True):
  """
  Build data for stan analysis. Similar to surv.build_parametric_data function,
  but simplified for the basic right-censored analysis.

  Parameters
  ----------
  ss_df : DataFrame
        DataFrame with "time_in_season" (float), "infection" (bool), 
        and "sample_date" (DateTime) columns at least.
  df : int
    Number of basis vectors in B-spline
  degree : int
    Default to cubic B-Spline
  include_intercept : bool
    Whether or not to include intercept in the B-spline projection

  Returns
  -------
  : dict
    Data for Stan fitting
  """

  time = ss_df.time_in_season.values
  time_trans = (time / np.max(time)) + 0.5 # Re-scaling time for easier estimation

  X, Xder = surv.get_bsplines(np.log(time_trans), df, degree, include_intercept)

  # Account for time chain rule
  time_mat = (np.tile(time_trans, Xder.shape[1])
                .reshape(Xder.shape[1], len(time_trans)).T)
  Xder = Xder / time_mat

  X_noint = X[:, 1:X.shape[1]]
  Xder_noint = Xder[:, 1:X.shape[1]]
  infected = ss_df.infection.astype(np.int).values

  standata = dict(N=len(time), X=X_noint, 
                  Xder=Xder_noint, p=X_noint.shape[1],
                  infected=infected)
  return(standata)


if __name__ == '__main__':

  # Some date preliminary parameters
  start_month = 1
  end_month = 12
  start = pd.datetime(2016, start_month, 1)
  end = pd.datetime(2016, end_month, 31)
  inf_days = pd.date_range(start, end)

  # Number of pigs in the entire region/population
  num_inf = 10000 

  # Number of pigs to sample from the full distribution
  num_pigs_array = [1000] #[100, 200, 300, 500, 1000]

  # Set up different time of infection probability distributions
  days = np.array([d.days for d in pd.to_datetime(inf_days) - start])
  unif = np.repeat(1 / (len(inf_days)), len(inf_days))
  unimod = surv.weibull_sampling_probs(days, 3, 200)
  bimod = surv.mixture_weibull_probs(days, [5, 10], [100, 250], [1, 1])
  all_probs = [unif, unimod, bimod]
  names = ["Uniform", "Unimodal", "Bimodal"]
  dfs = [4, 4, 11]

  pmod = 'stan_code/infectionrisk_rightcensored.pkl'

  if os.path.exists(pmod):
    smod = pd.read_pickle(pmod)
  else:
    smod = pystan.StanModel("stan_code/infectionrisk_rightcensored.stan")

  for num_pigs in num_pigs_array:
    print("Running analysis with {0} pigs...".format(num_pigs))

    fig, axes = plt.subplots(3, 3, figsize=(8, 8), sharex=True, sharey=True)

    # Loop over all combinations of distributions
    for i in range(len(all_probs)): # TOI distributions

      for j in range(len(all_probs)): # Sampling distributions

        ax = axes[i, j]
        toi_dates = np.random.choice(inf_days, size=num_inf, p=all_probs[i])
        sampling_days = np.random.choice(inf_days, size=num_pigs, p=all_probs[j])

        ss_df = surv.sample_pigs(sampling_days, toi_dates, start)
        standata = build_standata(ss_df, dfs[i])
        tfit = smod.sampling(standata, chains=2, iter=4000, warmup=2000)
        fitvals = surv.extract_fit(tfit, ss_df.time_in_season.values, dfs[i])

        # Custom KM estimate.
        km_est = surv.kaplan_meier(ss_df.time_in_season, ss_df.infection.astype(np.int))
        true_tsis = [d.days for d in pd.to_datetime(ss_df.toi) - pd.datetime(2016, start_month, 1)]
        full_tsis = [d.days for d in pd.to_datetime(toi_dates) - pd.datetime(2016, start_month, 1)]
        ss_df = ss_df.assign(true_tsi=true_tsis)
        inf_ecdf = surv.empirical_cdf(ss_df[ss_df.infection].true_tsi.values)
        true_ecdf = surv.empirical_cdf(full_tsis)

        # Plot results survival analysis.  Include KM estimate if desired.
        ax.plot(true_ecdf.data, true_ecdf.ecdf, lw=2, color="black", 
                                              label="Truth")

        ax.plot(inf_ecdf.data, inf_ecdf.ecdf, ls="--", color=sns.color_palette()[1],
                                              label="Not accounting\nfor censoring")

        median_logsurv = stats.scoreatpercentile(fitvals['logsurvival'], 50, axis=0)
        lower_logsurv = stats.scoreatpercentile(fitvals['logsurvival'], 2.5, axis=0)
        upper_logsurv = stats.scoreatpercentile(fitvals['logsurvival'], 97.5, axis=0)
        ax.plot(fitvals['newtime'], 1 - np.exp(median_logsurv), 
                                      color=sns.color_palette()[2],
                                      label="Accounting for\ncensoring")
        ax.fill_between(fitvals['newtime'], 1 - np.exp(upper_logsurv), 
                                            1 - np.exp(lower_logsurv), 
                                            color=sns.color_palette()[2],
                                            alpha=0.25)

        if ax.is_last_row() and j == 1:
          ax.set_xlabel("Days since start of infection season")

        if ax.is_first_col() and i == 1:
          ax.set_ylabel("Probability of infection by t, F(t)")

        if ax.is_first_row():
          ax.set_title(names[j], size=12)

        if ax.is_last_col():
          ax.text(1.1, 0.5, names[i], size=12, va="center", rotation=-90, transform=ax.transAxes)

        if i == 0 and j == 0:
          ax.legend(prop={'size': 6}, loc="lower right")


        ax.set_xticks(np.arange(0, 365, 75))

    # Add axis labels
    axes[0, 1].text(0.5, 1.3, "Sampling distribution", size=14, ha="center", 
                                                      transform=axes[0, 1].transAxes)
    axes[1, 2].text(1.3, 0.5, "True infection distribution", size=14, rotation=-90, 
                                         va="center", transform=axes[1, 2].transAxes)

    fig.savefig("../results/test_rightcensored_n{0}.pdf".format(num_pigs), bbox_inches="tight")
