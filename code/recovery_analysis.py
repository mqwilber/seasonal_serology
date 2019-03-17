## Python 3.6.8
import numpy as np # Version '1.14.2'
import survival as surv 
import pandas as pd # Version '0.23.4'
import scipy.stats as stats # Version '1.0.0'
import matplotlib.pyplot as plt # Version '2.2.2'
import seaborn as sns # Version '0.8.1'
import pystan # Version '2.18.1.0'
import os

"""
Description
-----------

Exploring the ability of a modified survival analysis to account for 
recovery in infection due to a seroconversion threshold. The script
below samples 1000 hosts (i.e. pigs) uniformly over a season and assigns
infection dates to each pig and associated antibody values based on a
known antibody curve and a seroconversion threshold. 

Author
-----
Mark Q. Wilber

"""

def pick_arm_of_curve(sero_df, tsifit):
  """ Get the TSI values on the correct arm of the curve """

  ind = sero_df.seropositive
  true_tsi = sero_df[ind].sample_date - sero_df[ind].true_toi
  true_tsi = np.array([t.days for t in true_tsi])

  maxg = surv.max_g(theta['A'], theta['r'], theta['B'], theta['d'], theta['X2'])
  deltas = stats.scoreatpercentile(tsifit.extract('deltas')['deltas'], 50, axis=0)
  est_tsi = deltas + maxg
  correct_tsi = np.array([surv.inverse_antibody_dynamics(stan_data['ab'][i], theta)[0] 
                                  if true_tsi[i] < maxg else est_tsi[i] 
                                  for i in range(T)])

  # Join these values back with the the sero_df data
  seropositive = sero_df[ind].seropositive
  correct_tsi_series = pd.Series(correct_tsi, index=seropositive.index, name="est_tsi")
  est_tsi_df = (pd.concat([seropositive, correct_tsi_series], axis=1)
                  .drop(columns=["seropositive"]))

  # Add estimated columns onto data frame
  sero_df_up = sero_df.join(est_tsi_df).fillna(-1)
  sero_df_up['est_toi'] = pd.NaT

  ind2 = ~sero_df_up.seropositive
  sero_df_up.loc[ind2, "est_toi"] = sero_df_up.sample_date[ind2] + pd.Timedelta(1, unit="D")
  sero_df_up.loc[~ind2, "est_toi"] = sero_df_up.sample_date[~ind2] - pd.to_timedelta(sero_df_up.est_tsi[~ind2], unit="D")

  return(sero_df_up)


def build_standata(est_df_surv, df, tau_falling, tau_rising):
  """
  A helper function that builds the the data used in the Stan analysis. Note that
  time is transformed to (time / maxtime) + 0.5 to help with estimation.
  """
  infected = est_df_surv.infection.astype(np.int).values
  time = est_df_surv.time_in_season.values
  maxtime = np.max(time)
  recovered = np.bitwise_and(infected == 0, time >= tau_falling).astype(np.int)

  mintime = np.min(time)
  nus = time - tau_falling
  nus[nus < mintime] = mintime 
  lower = time - tau_rising
  lower[lower < mintime] = mintime

  # Make B-splines
  degree = 3
  include_intercept = True
  trans_time = (time / maxtime) + 0.5
  X, Xder = surv.get_bsplines(np.log(trans_time), df, degree, include_intercept)
  Xder = surv.scale_X(Xder, trans_time)
  trans_nu = (nus / maxtime) + 0.5
  Xnu, _ = surv.predict_bspline(np.log(trans_nu), np.log(trans_time), df)
  trans_lower = (lower / maxtime) + 0.5
  Xlower, _ = surv.predict_bspline(np.log(trans_lower), np.log(trans_time), df)

  p = X.shape[1]
  standata = dict(N=X.shape[0], p=p - 1, X=X[:, 1:p], Xder=Xder[:, 1:p], 
                  infected=infected, recovery=recovered,
                  Xnu=Xnu[:, 1:p], Xlower=Xlower[:, 1:p])
  return(standata)

if __name__ == '__main__':

  # Some date preliminary parameters
  start_month = 5
  end_month = 4
  start = pd.datetime(2015, start_month, 1)
  end = pd.datetime(2016, end_month, 30)
  inf_days = pd.date_range(start, end)
  vals = np.arange(0, 365)

  # Parameters for the antibody curve and the cutoff
  theta = dict(X1=0.06, A=3.39, r=0.05, d=0.08, B=150, X2=0.03) # Parameters for the antibody curve
  sigma = 0.00 # No variability in the antibody curve
  tf = lambda x: np.log10(1 / x) # A transformation function
  cutoff = tf(0.673) # Seroconversion threshold
  tau_rising, tau_falling = surv.inverse_antibody_dynamics(cutoff, theta)

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
  dfs = [4, 4, 10]

  # Build stan model
  print("Building stan models...")
  weibull_recov = pystan.StanModel("stan_code/infectionrisk_recovery.stan")
  print("Completed building stan models")

  # Loop over all combinations of distributions
  for num_pigs in num_pigs_array:

    print("Working on {0} pigs sampled...".format(num_pigs))

    all_kms = []
    all_truth = []
    all_wrecov = []

    for i in range(3): # TOI distributions

      # Build serology data set
      sero_df = surv.build_serology_data(inf_days, all_probs[i], all_probs[0], 
                               num_pigs, theta, sigma, start,
                               total_pop=num_pop)
      sero_df = sero_df.assign(seropositive=lambda x: x.serodata > cutoff)

      # No uncertainty
      sind = sero_df.seropositive.values
      sero_df['toi'] = pd.NaT
      sero_df.loc[sind, 'toi'] = sero_df.true_toi[sind]
      sero_df.loc[~sind, 'toi'] = sero_df.sample_date[~sind] + pd.Timedelta(1, unit="D")
      est_df = sero_df[['sample_date', 'toi']]

      est_df_surv = surv.build_survival_data(est_df, start)

      # Build data and fit survival analysis with recovery 
      df = dfs[i]
      standata = build_standata(est_df_surv, df, tau_falling, tau_rising)
      wfit_recov = weibull_recov.sampling(data=standata, iter=4000, warmup=2000, chains=2)
      tsims = surv.extract_fit(wfit_recov, est_df_surv.time_in_season.values, df)
      all_wrecov.append(tsims)

      # Fit Kaplan-Meier estimates without accounting for recovery for comparison
      km = surv.kaplan_meier(est_df_surv.time_in_season, 
                             est_df_surv.infection.values.astype(np.int))
      all_kms.append(km)

      # Get the truth 
      true_tsi = np.array([d.days for d in  sero_df.true_toi - start])
      true_ecdf = surv.empirical_cdf(true_tsi)
      all_truth.append(true_ecdf)


    # Plot the simulation results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, ax in enumerate(axes):

      true_ecdf = all_truth[i]
      km = all_kms[i]
      fitvals = all_wrecov[i]

      # Extract the survival estimates
      median_wei = 1 - np.exp(stats.scoreatpercentile(fitvals['logsurvival'], 50, axis=0))
      lower_wei = 1 - np.exp(stats.scoreatpercentile(fitvals['logsurvival'], 2.5, axis=0))
      upper_wei = 1 - np.exp(stats.scoreatpercentile(fitvals['logsurvival'], 97.5, axis=0))
      newtime = fitvals['newtime']

      ax.plot(true_ecdf.data, true_ecdf.ecdf, lw=2, color="black", label="Truth")

      ax.plot(km.index, 1 - km.survival, ls="--", label="Not accounting for\nrecovery", 
                                          color=sns.color_palette()[1])

      ax.plot(newtime, median_wei, label="Accounting for recovery",
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
      fig.savefig("../results/test_recovery_n{0}.pdf".format(num_pigs), bbox_inches="tight")



