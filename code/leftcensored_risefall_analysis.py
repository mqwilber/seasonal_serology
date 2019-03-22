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
left-censoring due to a seroconversion threshold and the rise
and fall of the antibody curve.

Author
------
Mark Q. Wilber

"""

def build_rising_falling_data(sero_df, tau_end, theta, yend, leftcensored):
  """ Build survival datasets for both rising and falling tsis """

  # Get the rise and fall TOI
  newcols = ['toi_rising', 'toi_falling']
  infected = sero_df.seropositive
  rfdata = [surv.inverse_antibody_dynamics(ab, theta) for ab in sero_df.serodata]
  rfvects = list(zip(*rfdata))

  for i in range(len(newcols)):
      
    nc = newcols[i]
    rf = np.array(rfvects[i])
    sero_df[nc] = pd.NaT
    
    if i == 0: # Rising
      sind = (sero_df.seropositive).values
      sero_df[nc] = pd.NaT
      sero_df.loc[sind, nc] = sero_df.sample_date[sind] -  pd.to_timedelta(rf[sind], unit="D")
      sero_df.loc[~sind, nc] = sero_df.sample_date[~sind] + pd.Timedelta(1, unit="D")
    else: # Falling
      sind1 = (sero_df.seropositive & ~leftcensored).values
      sind2 = (sero_df.seropositive &  leftcensored).values
      sind3 = (~sero_df.seropositive).values
      sero_df[nc] = pd.NaT
      sero_df.loc[sind1, nc] = sero_df.sample_date[sind1] - pd.to_timedelta(rf[sind1], unit="D")
      sero_df.loc[sind2, nc] = sero_df.sample_date[sind2]
      sero_df.loc[sind3, nc] = sero_df.sample_date[~sind3] + pd.Timedelta(1, unit="D")
          
  est_rising = sero_df[['sample_date', 'toi_rising', "serodata"]]
  est_rising = est_rising.rename(columns={'toi_rising': 'toi'})
  est_falling = sero_df[['sample_date', 'toi_falling', "serodata"]]
  est_falling = est_falling.rename(columns={'toi_falling': 'toi'})

  surv_rising = surv.build_survival_data(est_rising, start)
  surv_falling = surv.build_survival_data(est_falling, start)

  return((surv_falling, surv_rising, leftcensored))

if __name__ == '__main__':

  # Some date preliminary parameters
  start_month = 5
  end_month = 4
  start = pd.datetime(2015, start_month, 1)
  end = pd.datetime(2016, end_month, 30)
  start_season = start 
  end_season = end
  inf_days = pd.date_range(start, end)
  vals = np.arange(0, 365)

  # Parameters for the antibody curve and the cutoff
  theta = dict(X1=0.06, A=3.39, r=0.05, d=0.08, B=150, X2=0.13)
  sigma = 0.00 # No uncertainty in the antibody curve...in this simulation
  tf = lambda x: np.log10(1 / x)
  cutoff = tf(0.85)
  rtol = 0.01 # Buffer around left-censoring
  tau_rising, tau_falling, tau_max = surv.get_abcurve_properties(theta, cutoff)
  tau_end = surv.get_tau_end(theta, rtol=rtol)
  tau_min = surv.inverse_antibody_dynamics(0.19, theta)[0]
  yend = surv.antibody_dynamics(tau_end, theta, 0)[0]

  # Number of pigs in the entire region/population
  num_pop = 10000 

  # Number of pigs to sample from the full distribution
  num_pigs_array = [1000]

  # Build the different incidence distributions
  days = np.array([d.days for d in pd.to_datetime(inf_days) - start])
  unif = np.repeat(1 / (len(inf_days)), len(inf_days))
  unimod = surv.weibull_sampling_probs(days, 3, 200)
  bimod = surv.mixture_weibull_probs(days, [5, 10], [100, 250], [1, 1])
  all_probs = [unif, unimod, bimod, unif, unimod, bimod, unif, unimod, bimod]
  names = ["Uniform", "Unimodal", "Bimodal"]
  dfs = [4, 4, 14]
  degree = 3
  include_intercept = True

  # Build stan model
  print("Building stan models...")
  lcmods = []
  nms = ["stan_code/infectionrisk_leftcensored_rf.pkl",
         "stan_code/infectionrisk_leftcensored.pkl"]

  for nm in nms:

    if os.path.exists(nm):
      lcmod = pd.read_pickle(nm)
    else:
      lcmod = pystan.StanModel(nm.split(".")[0] + ".stan")
      pd.to_pickle(lcmod, nm)

    lcmods.append(lcmod)

  print("Completed building stan models")

  # Loop over all combinations of distributions
  for num_pigs in num_pigs_array:

    all_truth = []
    all_wei = []
    all_falling = []

    for i in range(3): # TOI distributions

      # Build serology data set
      inf_probs = all_probs[i]
      sero_df = surv.build_serology_data(inf_days, inf_probs, all_probs[0], 
                                         num_pigs, theta, sigma, start,
                                         total_pop=num_pop)
      sero_df = sero_df.assign(seropositive=lambda x: x.serodata > cutoff)
      leftcensored = (((sero_df.serodata < yend) & 
                       (sero_df.serodata >= theta['X1'] + theta['X2'])) &
                        sero_df.seropositive)

      res = build_rising_falling_data(sero_df, tau_end, theta, yend, leftcensored)
      surv_falling, surv_rising, leftcensored = res

      # Set up data for serology analysis
      recovered = np.repeat(0, len(surv_rising)) 
      leftcensored = leftcensored.values.astype(np.int)
      surv_df = surv_falling
      surv_df = surv_df.rename(columns={'time_in_season':"time_in_season_falling"})
      surv_df['time_in_season_rising'] = surv_rising.time_in_season
      df = dfs[i]
      standata = surv.build_parametric_data(surv_df, recovered, leftcensored, 
                                theta, start, df, 
                                degree=degree, include_intercept=include_intercept, 
                                cutoff=cutoff, use_tau_end=True, calc_dob=False,
                                rtol=rtol, yend=yend, tau_min=tau_min)

      surv_df = surv_df.rename(columns={'time_in_season_falling':"time_in_season"})
      standata_falling = surv.build_parametric_data(surv_df, recovered, leftcensored, 
                                theta, start, df, 
                                degree=degree, include_intercept=include_intercept, 
                                cutoff=cutoff, use_tau_end=False, calc_dob=False,
                                rtol=rtol, yend=yend, tau_min=None)

      # Fit the stan model
      def init_df():
        init_vals = {}
        init_vals['intercept'] = -5
        betas = np.linspace(2, 6, num=df - 1)
        init_vals['betas'] = betas
        return(init_vals)

      # Fit model with rise and fall
      lcfit = lcmods[0].sampling(standata, iter=4000, warmup=2000, chains=2, init=init_df)
      time = surv_df.time_in_season.values
      curves = surv.extract_fit(lcfit, time, df)
      all_wei.append(curves)

      # Fit model with only fall
      lcfit = lcmods[1].sampling(standata_falling, iter=4000, warmup=2000, chains=2, init=init_df)
      time = surv_df.time_in_season.values
      curves = surv.extract_fit(lcfit, time, df)
      all_falling.append(curves)

      ## Get the truth
      toi_dates = np.random.choice(inf_days, size=num_pop,
                                      replace=True, 
                                      p=inf_probs)
      true_tsi = np.array([d.days for d in pd.to_datetime(toi_dates) - start_season])
      true_ecdf = surv.empirical_cdf(true_tsi)
      all_truth.append((true_ecdf, true_tsi))

    # Plot the results of the analysis
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes = axes.ravel()
    colors = sns.color_palette()

    for i, ax in enumerate(axes):

      true_ecdf, true_vals = all_truth[i]
      curves_rf = all_wei[i]
      Frf = 1 - np.exp(stats.scoreatpercentile(curves_rf['logsurvival'], 50, axis=0))
      Frf_upper = 1 - np.exp(stats.scoreatpercentile(curves_rf['logsurvival'], 2.5, axis=0))
      Frf_lower = 1 - np.exp(stats.scoreatpercentile(curves_rf['logsurvival'], 97.5, axis=0))

      curves_f = all_falling[i]
      Ff = 1 - np.exp(stats.scoreatpercentile(curves_f['logsurvival'], 50, axis=0))
      Ff_upper = 1 - np.exp(stats.scoreatpercentile(curves_f['logsurvival'], 2.5, axis=0))
      Ff_lower = 1 - np.exp(stats.scoreatpercentile(curves_f['logsurvival'], 97.5, axis=0))


      ax.plot(true_ecdf.data, true_ecdf.ecdf, color="black", lw=2, label="Truth")
      ax.plot(curves['newtime'], Frf, color=colors[2], label="Accounting for\nrising and falling")
      ax.fill_between(curves['newtime'], Frf_lower, Frf_upper, alpha=0.2, color=colors[2])

      ax.plot(curves['newtime'], Ff, color=colors[1], ls="--", label="Only falling")
      ax.fill_between(curves['newtime'], Ff_lower, Ff_upper, alpha=0.2, color=colors[1])

      if ax.is_first_col():
        ax.set_ylabel("Probability of infection by t, F(t)", size=15) 
        ax.legend(prop={'size': 9})

      if i == 1:
        ax.set_xlabel("Days since start of infection season", size=15)
             
      ax.set_title(names[i], size=15)

    fig.savefig("../results/test_leftcensored_risefall_n{0}.pdf".format(num_pigs), bbox_inches="tight")

