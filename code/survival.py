## Python 3.6.8
import numpy as np # Version '1.14.2'
import scipy.stats as stats  # Version '1.0.0'
import pandas as pd # Version '0.23.4'
from scipy.interpolate import interp1d # Version '1.0.0'
from scipy import interp # Version '1.0.0'
from scipy.optimize import brentq, fmin # Version '1.0.0'
from calendar import monthrange
from patsy import dmatrix, bs, build_design_matrices # Version '2.18.1.0'
from patsy.splines import BS # Version '2.18.1.0'
import statsmodels.api as sm # Version '0.8.0'

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def get_abcurve_properties(theta, threshold, maxtime=500):
  """
  Given the parameters of an antibody curve and a seroconversion threshold,
  compute tau_rising, tau_falling, and maxg

  Parameters
  ----------
  theta : dict
    Dictionary holding the six parameters for the antibody curve
      'X1': Baseline antibody level prior to exposure
      'A': The initial lag between exposure and antibody production
      'r': Antibody production rate
      'B': Period of antibody production in response to infection
      'd': Antibody decay rate
      'X2': Baseline antibody levels following antibody decay
  threshold : float
    The seroconversion threshold

  Returns
  -------
  : tuple
    (tau_rising, tau_falling, maxg)

  """

  setdiff = set(["X1", "A", "r", "B", "d", "X2"]) - set(theta.keys())
  assert len(setdiff) == 0, "Missing parameter(s): {0}".format(setdiff)

  tau_rising, tau_falling = inverse_antibody_dynamics(threshold, theta, maxtime=maxtime)
  maxg = max_g(theta['A'], theta['r'], theta['B'], theta['d'], theta['X2'])
  return((tau_rising, tau_falling, maxg))


def antibody_dynamics(delta, theta, sigma, c=0, sigma_below=None):
  """
  The antibody curve given in Pepin et al. 2017 (Ecology Letters)
  
  Parameters
  ----------
  delta : float
    Time since infection 
  theta : dict
    Dictionary holding the six parameters
      'X1': Baseline antibody level prior to exposure
      'A': The initial lag between exposure and antibody production
      'r': Antibody production rate
      'B': Period of antibody production in response to infection
      'd': Antibody decay rate
      'X2': Baseline antibody levels following antibody decay
    sigma : float
      Standard deviation of measurement error in anti-body data.
    c : float
      factor that allows sigma to vary with time

  Returns
  -------
  ab : float
    The antibody level at time delta
  """

  t = Bunch(theta) # for clean access

  if delta < t.A:

    ab = t.X1

  elif (delta >= t.A) and (delta < (t.A + t.B)):

    dA = delta - t.A
    ab = t.X1 + t.X2*(dA / t.B) + \
         (t.r + t.d*t.r*(t.B - dA)) / (t.B * t.d**2) - \
         ((t.r + t.d*t.r*t.B) / (t.B * t.d**2)) * np.exp(-t.d * dA)
  else:

    dA = delta - t.A
    ab = t.X1 + t.X2 + \
         ((t.r*np.exp(t.d*t.B) - t.r - t.d*t.r*t.B) / (t.B*t.d**2))*np.exp(-t.d*dA) 

  
  return(ab + stats.norm.rvs(0, np.sqrt(sigma**2*np.exp(2*c*delta)), size=1))


def max_g(A, r, B, d, X2):
  """
  Computes the time at which the antibody curve is maximized

  Parameters
  ---------- 
  A : float 
    The initial lag between exposure and antibody production
  r : float
    Antibody production rate
  B : float 
    Period of antibody production in response to infection
  d : float
    Antibody decay rate
  X2 : float 
    Baseline antibody levels following antibody decay

  Returns
  -------
  : float
    The time at which the antibody curve is maximized
  """

  maxt = (1 / -d)*(np.log((r / (B*d)) - X2 / B) - (np.log((r + d*r*B) / (B*d)) + d*A))
  return(maxt)

def get_tau_end(theta, maxtime=500, rtol=0.01):
  """
  Get the time at which the antibody curve flattens out

  Parameters
  ---------- 
  theta : dict
    Dictionary of parameters. Same as antibody_dynamics

  Returns
  -------
  : float
    Time at which antibody curve levels off after rise.
    In other words, approximately equals X1 + X2
  """

  Xend = theta['X1'] + theta['X2']
  maxg = max_g(theta['A'], theta['r'], theta['B'], theta['d'], theta['X2'])
  time = np.linspace(0, maxtime, num=1000)
  abvals = np.array([antibody_dynamics(d, theta, 0)[0] for d in time])
  ind_end = np.isclose(abvals, Xend, rtol=rtol) & (time > maxg)
  argtau_end = np.where(ind_end)[0].min()
  tau_end = time[argtau_end]
  return(tau_end)


def inverse_antibody_dynamics(ab, theta, maxtime=500):
  """
  Given an antibody level, get the two possible time since infections

  Parameters
  ----------
  ab : float
    Antibody level
  theta : dict
    Dictionary of parameters. Same as antibody_dynamics
  maxtime : float
    Maximum time since infection to consider

  Returns
  -------
  : tuple
    (lower time is ab level, upper time with ab level)

  """

  maxg = max_g(theta['A'], theta['r'], theta['B'], theta['d'], theta['X2'])

  zeroroot = lambda x: antibody_dynamics(x, theta, 0) - ab

  try:
    lower = brentq(zeroroot, -1, maxg)
  except:

    if np.round(antibody_dynamics(maxg, theta, 0), 1) == np.round(ab, decimals=1): 
      lower = maxg
    else:
      lower = theta['A']

  try:
    upper = brentq(zeroroot, maxg, maxtime)
  except:
    upper = maxtime

  return((lower, upper))


def kaplan_meier(duration, event_observed):
  """
  Implementation of the standard Kaplan-Meier estimator

  Parameters
  ----------
  duration : array-like
    The duration to an event or right censoring
  event_observed : array-like
    Boolean array where 1 indicates the event happened and 0 indicates 
    right-censoring

  Returns
  -------
  : DataFrame with index of unique event times
      at_risk : The number at risk at a time
      observed : The number of events observed at the time
      survival : The estimated survival probability at the time

  """

  non_censored = duration[event_observed.astype(np.bool)]
  ts = np.sort(non_censored)
  unq_ts = np.r_[[0], np.unique(ts)]

  ds = np.empty(len(unq_ts))
  Ys = np.empty(len(unq_ts))

  # Calculate at-risk individuals
  for i, t in enumerate(unq_ts):

    Ys[i] = np.sum(duration >= t)
    ds[i] = np.sum(ts == t)

  # Calculate survival function
  surv_ratios = (1 - (ds / Ys))
  hazard = ds / Ys
  surv_function = np.cumprod(surv_ratios)

  return(pd.DataFrame(dict(at_risk=Ys, 
                           observed=ds, 
                           survival=surv_function,
                           hazard=hazard), 
                      index=unq_ts))


def sample_pigs(sampling_days, toi_dates, start_season, dob_dates=None):
    """
    Generate a survival analysis dataframe from a sampling scheme and 
    pig infection dates, generate a survival analysis. Pigs is can be generically
    thought of as hosts. 

    Parameters
    ----------
    sampling_days : array
      Each item is a date at which a sampling event occurred
    toi_dates : array
      Each entry represents an individual with their date (time) of infection
    start_season : pd.Datetime value or array
      The baseline date from which time of infection should be calculated
    dob_dates : array-like or None
      Also pass in date of birth dates to be sampled

    Returns
    -------
    : DataFrame
      DataFrame formatted for survival analysis
    """
    
    samp_stats = pd.Series(sampling_days).value_counts()

    # Sample pigs by days
    serology_samples = {}
    for samptime in samp_stats.index:

      if dob_dates is None:
        samp_ind = np.random.choice(range(len(toi_dates)), 
                                          size=samp_stats[samptime])
        toi = toi_dates[samp_ind]
        serology_samples[samptime] = pd.DataFrame({'toi': np.atleast_1d(toi)})

      else:

        # Only sample pigs that are in existence 
        avail_ind = (pd.Series(dob_dates) <= samptime).values
        samp_ind = np.random.choice(range(len(toi_dates[avail_ind])),
                                              size=samp_stats[samptime])
        toi = toi_dates[avail_ind][samp_ind]
        dob = dob_dates[avail_ind][samp_ind]

        serology_samples[samptime] = pd.DataFrame({'toi': np.atleast_1d(toi),
                                                   'dob': np.atleast_1d(dob)})

    ss_df = pd.concat(serology_samples)
    ss_df = (ss_df.reset_index()
                  .drop(columns=['level_1'])
                  .rename(columns=dict(level_0="sample_date")))

    # Time since the start of the season at which either an infection occurs
    # or sampling occurs (censored)
    ss_df = build_survival_data(ss_df, start_season)
    
    return(ss_df)


def build_survival_data(ss_df, start_season):
  """
  Format data for survival analysis

  Parameters
  ----------

  ss_df : DataFrame
    ss_df must have columns 
      - `toi`: pd.DateTime objects giving the time of infection. If no infection was
         recorded, these should be larger than sample_date
      - `sample_date`: pd.DateTime objects specify the date that the sample was
         taken
    Can also have columns 'dob' for date of birth.
  start_season : pd.datetime, single value or Series of len(ss_df)
    The start of the infection season for each sample, or a single start for
    all samples

  Returns
  -------
  : DataFrame for survival analysis
    - `time_in_season`: The duration variable in a survival analysis
    - `infection`: Whether or not a infection was observed. The censoring 
       variable in a survival analysis.
  """

  if "dob" in list(ss_df.columns):
    infection = (ss_df.toi <= ss_df.sample_date) & (ss_df.dob <= ss_df.toi)
  else:
    infection = (ss_df.toi <= ss_df.sample_date) 

  ss_df = ss_df.assign(infection=infection)

  ss_df['census_time'] = pd.NaT
  ss_df.loc[~ss_df.infection, "census_time"] = ss_df.sample_date[~ss_df.infection]
  ss_df.loc[ss_df.infection, "census_time"] = ss_df.toi[ss_df.infection]

  # Time since start of infection season
  time_in_season = np.array([x.days for x in ss_df.census_time - start_season])

  ss_df = ss_df.assign(time_in_season=time_in_season)

  if len(np.atleast_1d(start_season)) == 1:
    years = start_season.year
  else:
    years = start_season.dt.year

  ss_df = ss_df.assign(infection_season=years)

  # Make zeros and less than zero time close to 0, but not quite.
  ss_df.loc[ss_df.time_in_season <= 0, 'time_in_season'] = 0

  return(ss_df)

def extract_theta(fit, parnames=['X2', 'B', 'X1', 'A', 'r', 'd'], 
                       known_params={}, median=False):
  """
  Make a random theta parameter dict from stan fit object

  Parameters
  ----------
  fit : Pystan model from tsi_model_contrained_minmax.stan 
  parnames : list
    List of parameter names to extract from the model
  known_params : dict
    Dictionary with parameter names and known values

  Returns
  -------
  : dict
    Parameters for antibody curve

  """

  num = len(fit[parnames[0]])

  samp = np.random.randint(0, num, size=1)

  if median:
    theta_samp = {par : np.median(fit[par]) for par in parnames}
  else:
    theta_samp = {par : fit[par][samp][0] for par in parnames}

  theta_samp.update(known_params)

  return((theta_samp, samp))

def compute_rising_falling_probs(tau_rising, tau_falling, tau_max, 
                                 sample_date, start_date, leftcensored=False,
                                 tau_min=None):
  """
  Compute the probability that a sample came from either the rising or falling
  portion of the antibody curve. Depends on when a sample was taken and the start
  date/DOB of the sample of interest. 

  Parameters
  ----------
  tau_rising : float
    Time rising arm crosses the seroconversion threshold.
  tau_falling : float
    Time falling arm crosses the seroconversion threshold
  tau_max : float
    Time at which antibody curve is maximized. Tau_max in paper. 
  sample_data : pd.datetime
    The date that the sample was taken
  start_date : pd.datetime
    The start of the infection season
  leftcensored : bool
    If False, assumes that point for which the probabilities are being computed 
    is not left-censored. If True, assumes the point is left-censored and uses
    a different algorithm for computing the rising and falling probability.
  tau_min : float
    If considering leftcensored, tau_min is true time at which the the rising
    arm of the antibody curve crosses the leftcensoring threshold. In practice, 
    one might add a slight buffer around

  Returns
  -------
  : tuple
    (probability sample on rising arm, probability sample on falling arm)

  Notes
  -----
  Rounding time to days.

  """

  # Compute toi_min, toi_max, tau_upper_date
  if tau_min == None:
    tau_min = tau_rising

  if not leftcensored:

    tau_upper_date = start_date + pd.Timedelta(tau_falling, unit="D")

    if (sample_date >= tau_upper_date):

      prising = (tau_max - tau_rising) / (tau_falling - tau_rising)

    else: 

      samp_delta = (sample_date - start_date).days
      prising = np.min([1, (tau_max - tau_rising) / (samp_delta - tau_rising)])

  else:

    assert tau_rising >= tau_min, "tau_rising must be greater than or equal to tau_min"

    samp_delta = (sample_date - start_date).days
    prising = (tau_rising - tau_min) / (tau_rising - tau_min + samp_delta - tau_falling)

  pfalling = 1 - prising
  return((prising, pfalling)) 


def infection_period(date, start_month=8, end_month=7):
  """
  Determines the seasonal window that date falls into.

  Parameters
  ----------
  date : pandas TimeStamp
    Date to be analyzed
  start_month : int
    Month number that starts the infection period
  end_month : int
    Month number that ends the infection period
  
  Returns
  -------
  : tuple of TimeDelta, TimeStamp, TimeStamp
    (Time since start of seasonal period, start of season, end of season)
  
  """
  
  # Start season
  if date.month in range(1, start_month):
      start_season = pd.datetime(date.year - 1, start_month, 1)
      end_season = pd.datetime(date.year, end_month, 
                                monthrange(date.year, end_month)[1])
  else:
      start_season = pd.datetime(date.year, start_month, 1)
      end_season = pd.datetime(date.year + 1, end_month,
                               monthrange(date.year + 1, end_month)[1])
  
  return((date - start_season, start_season, end_season))
    

def normal_sampling_probs(dates, mu=0, sigma=1):
  """ Weight dates via a normal distribution """
  
  st = pd.Series(dates)
  jdate = [s.to_julian_date() for s in st]
  std_jdate = (jdate  - np.mean(jdate)) / np.std(jdate)

  # Sample according to a normal distribution
  weight_probs = stats.norm.pdf(std_jdate, loc=mu, scale=sigma)

  probs = weight_probs / np.sum(weight_probs)
  return(probs)

def weibull_sampling_probs(days_since_start, alpha, sigma):
  """ Get Weibull sampling probs """

  # Sample according to a normal distribution
  weight_probs = weibull_pdf(days_since_start, alpha, sigma)

  probs = weight_probs / np.sum(weight_probs)
  return(probs)

def mixture_weibull_probs (days_since_start, alphas, sigmas, weights):
  """ A mixture weibull distribution

  Parameters
  ----------
  alphas : array-like
    Inverse scale (smaller is more spread)
  sigmas : array-like
    Location parameters
  weights : array-like
    How much weight to give each distribution

  """
  all_probs = [weights[i]*weibull_sampling_probs(days_since_start, alphas[i], 
                                                  sigmas[i]) for
                  i in range(len(alphas))]
  totprobs = np.array(all_probs).sum(axis=0)
  probs = totprobs / np.sum(totprobs)
  return(probs)

def mixture_sampling_probs(dates, mus, sigmas, weights):
  """ Weight dates via a mixture normal distribution """
  
  st = pd.Series(dates)
  jdate = [s.to_julian_date() for s in st]
  std_jdate = (jdate  - np.mean(jdate)) / np.std(jdate)

  # Sample according to a normal distribution
  weight_vects = [weight * stats.norm.pdf(std_jdate, 
                                          loc=mu, 
                                          scale=sigma)[:, np.newaxis] 
                      for mu, sigma, weight in zip(mus, sigmas, weights)]

  weight_probs = np.hstack(weight_vects).sum(axis=1)
  probs = weight_probs / np.sum(weight_probs)

  return(probs)


def build_serology_data(dates, infection_probs, sampling_probs, 
                    numpigs, theta, sigma, start_season, 
                    total_pop=10000, prev_buffer=None, dob_dates=None):
  """ 
  Sample hosts as if we were taking serology samples from the field

  Parameters
  ----------
  dates : array-like
    Array of dates by day
  infection_probs : array-like
    Probability of infection for each day in dates
  sampling_probs : array-like
    Probability of sampling for each day in dates
  numpigs : int
    Number of pigs to sample. Less or equal to total_pop
  theta : dict
    Dictionary of parameters that define the antibody curve
  sigma : float
    Variability in the antibody curve
  start_season : pd.datetime
    The date considered the start of the infection season
  prev_buffer : float
    Between 0 and 1. Specify the true prevalence (not seroprevalence) at the end
    of the infection season. Default is 1.

  Returns
  --------
  : DataFrame
  """

  toi_dates = np.random.choice(dates, size=total_pop,
                                      replace=True, 
                                      p=infection_probs)

  # Reduce true prevalence
  if prev_buffer != None:

      inds = np.random.choice(np.arange(total_pop), replace=False, 
          size=np.int(np.floor((1 - prev_buffer)*total_pop)))
      toi_dates[inds] = pd.to_datetime(np.max(dates)) + pd.Timedelta(30, unit="D")

  sampling_days = np.random.choice(dates, size=numpigs, p=sampling_probs)
  ss_df = sample_pigs(sampling_days, toi_dates, start_season, dob_dates=dob_dates)

  # Get serology at sampling
  tsi = ss_df.sample_date - ss_df.toi 
  tsi_days = [t.days if t.days > 0 else 0 for t in tsi]

  serodata = np.array([antibody_dynamics(x, theta, sigma)[0] for x in tsi_days])

  sero_df = pd.DataFrame(dict(sample_date=ss_df.sample_date, 
                              serodata=serodata,
                              true_toi=ss_df.toi))
  if dob_dates is not None:
    sero_df['dob'] = ss_df.dob


  return(sero_df)


def build_stan_data(lab_data, field_data, raw_threshold=0.673):
  """
  Specific preparations for lab and field data

  Parameters
  ----------
  lab_data : DataFrame
    The influenza lab data from Sun et al.
  field_data : DataFrame
    The field data
  raw_threshold : float
    The unconverted threshold S:N threshold. Default based on USDA data

  Returns
  -------
  : dict
    Dict contains necessary data for fitting stan model

  """

  # Format the lab data. 
  # Shifting the Sentinel's time series so we can use them when fitting the 
  # model.
  ed = lab_data[lab_data.day <= 93]
  val = ed.loc[ed.treatment == "S", "day"] - 5
  ed.loc[ed.treatment == "S", "day"] = val
  ed = ed[ed.day >= 0]

  tf = lambda x: np.log10(1 / x)
  threshold = tf(raw_threshold)

  # Extract and format the field data
  abvals = tf(field_data[~np.isnan(field_data.sn_ratio)].sn_ratio.values)
  num_positive = np.sum(abvals >= threshold)
  positive_id = np.repeat(-1, len(abvals))
  positive_id[abvals >= threshold] = np.arange(1, num_positive + 1)

  # Make the data for stan
  standata = {'N' : len(ed),
              'antibody': tf(ed.SN).values,
              'time': ed.day.values,
              'T': len(abvals),
              'ab': abvals,
              'X2': 0.03,
              'B': 150,
              'threshold' : threshold, 
              'P' : num_positive,
              'positive_id' : positive_id}

  return(standata)

def calc_tsi(fit, samples, fixed_params={}, showprogess=True):
  """
  Calculate the time since infection distributions from the stan object

  Parameters
  ----------
  fit : Fitted stan object from tsi_model_contrained*.stan
  fixed_params : dict
    Keys are the antibody parameters that are fixed when estimating TSI
    Values are the values of those parameters.
  samples : int
    How many samples to take from the posterior. Should be less than the number
    of MCMC samples in the posterior
  showprogress : bool
    Can be a bit slow so setting to True shows the progress

  Returns
  -------
  : tuple
    tsi_min : The estimated time since infection on the rising arm
    tsi_max : The estimated time since infection on the falling arm
    inds : The indexes for the sample taken from the posterior in fit
  """

  all_params = ['A', 'B', 'X1', 'X2', 'r', 'd']
  non_fixed_params = list(set(all_params) - set(list(fixed_params.keys())))

  num = fit['taus'].shape[0]
  inds = np.random.randint(0, num, size=samples)
  num_pos = fit['taus'].shape[1]

  samp_params = {}
  for p in non_fixed_params:
    samp_params[p] = fit[p][inds]

  for p in fixed_params.keys():
    samp_params[p] = np.repeat(fixed_params[p], len(inds))

  dmaxs = fit['taus'][inds, :]

  # Calculate maximum time since infection
  maxg = np.array([max_g(A, r, B, d, X2) for A, r, B, d, X2 in zip(samp_params['A'], 
                                                            samp_params['r'],
                                                            samp_params['B'],
                                                            samp_params['d'],
                                                            samp_params['X2'])])
  tsi_max = np.tile(maxg[:, np.newaxis], num_pos) + dmaxs

  # Get antibody values
  tsi_min = np.empty((samples, num_pos))
  for i in range(num_pos):

    if showprogess:
      if (i + 1) % 10 == 0:
        print("Sample {0} of {1}".format(i + 1, num_pos))

    ab_vals = np.array([antibody_dynamics(t, dict(X1=X1, 
                                                  A=A, 
                                                  r=r, 
                                                  B=B, 
                                                  d=d, 
                                                  X2=X2), 0)[0]
                      for t, A, r, B, d, X2, X1 in zip(tsi_max[:, i], 
                                                            samp_params['A'], 
                                                            samp_params['r'],
                                                            samp_params['B'],
                                                            samp_params['d'],
                                                            samp_params['X2'],
                                                            samp_params['X1'])])

    # Invert and get early time
    tmin = np.array([inverse_antibody_dynamics(a, dict(X1=X1, 
                                                         A=A, 
                                                         r=r, 
                                                         B=B, 
                                                         d=d, 
                                                         X2=X2), 0)[0]
                      for a, A, r, B, d, X2, X1 in zip(ab_vals, samp_params['A'], 
                                                            samp_params['r'],
                                                            samp_params['B'],
                                                            samp_params['d'],
                                                            samp_params['X2'],
                                                            samp_params['X1'])])
    tsi_min[:, i] = tmin

  return((tsi_min, tsi_max, inds))

def empirical_cdf(data):
  """
  Generates an empirical cdf from data

  Parameters
  ----------
  data : iterable
    Empirical data

  Returns
  --------
  DataFrame
    Columns 'data' and 'ecdf'. 'data' contains ordered data and 'ecdf'
    contains the corresponding ecdf values for the data.

  """

  vals = pd.Series(data).value_counts()
  ecdf = pd.DataFrame(data).set_index(keys=0)
  probs = pd.DataFrame(vals.sort_index().cumsum() / np.float(len(data)))
  ecdf = ecdf.join(probs)
  ecdf.index.name = "data"
  ecdf = ecdf.reset_index()
  ecdf.columns = ['data', 'ecdf']
  ecdf = ecdf.sort_values(by="ecdf")

  return(ecdf)

def weibull_pdf(x, alpha, sigma):
    pdf = (alpha / sigma) * (x / sigma)**(alpha - 1) * np.exp(-(x / sigma)**alpha)
    return(pdf)

def weibull_cdf(x, alpha, sigma):
    cdf = 1 - np.exp(-(x / sigma)**alpha)
    return(cdf)

def weibull_hazard(vals, alpha, sigma):
    lam = (1 / sigma**alpha)
    hazard = lam * alpha * vals**(alpha - 1)
    return(hazard)

def ma_derivative(x, y, steps=3):
    """
    Compute the numerical derivative for x and y using differences

    Parameters
    ----------
    x : array-like
    y : array-like
    steps : int
      The step size (number of indexes in x and y) or which to compute slope

    Returns
    -------
    : tuple
      (midpoint over which the derivative was calculated, derivatives)
    """

    h = x[steps] - x[0]
    y1 = y[:(len(y) - steps)]
    y2 = y[steps:]

    x1 = x[:(len(x) - steps)]
    x2 = x[steps:]
    midpoints = (x1 + x2) / 2

    deriv = (y2 - y1) / h

    return((midpoints, deriv))

def coerce_index(factor, lookup):
  """
  Convert factor Series into numbers. Similar to the the rethinking package. 
  Useful for fitting random effects.

  Parameters
  ----------
  factor : Series object
    The Series object that will be converted.
  lookup : Series object
    Will use lookup vector to build lookup table. Otherwise uses factor. Just
    pass factor twice if you want to use this as the lookup

  Returns
  -------
  : Series
    factor converted to unique numeric ids.

  """

  lookuptab = pd.Series(lookup.unique()).reset_index()
  lookuptab.columns = ['idnum', factor.name]
  coerced = factor.reset_index().merge(lookuptab, on=factor.name, how="left").set_index(factor.name)

  return((coerced.idnum, lookuptab))


def fit_bspline(x, y, df, degree):
  """
  Fit a Bspline to survival curve and get derivatives

  Parameters
  ----------
  x : array-like
    Time since start of season values
  y : array-like
    Probabilities from 0 to 1. 1 - S(t)
  df : float
    Degree of freedom for the basis function. Number of basis vectors
  degree : float
    Use 3 for a cubic Bspline.

  Returns
  -------
  : tuple
    (x, predicted y at x, predicted first derivatives at x)

  """

  X = dmatrix("bs(x, df={0}, degree={1}, include_intercept=True) - 1".format(df, degree), 
                            {"x": x}, return_type="dataframe")

  # Fit Bspline
  fit = sm.GLM(y, X).fit()
  pred_vals = fit.predict(X).values

  # Get the knots for computing the first derivative
  bsobj = BS()
  bsobj.memorize_chunk(x, df=df, degree=degree, include_intercept=True)
  bsobj.memorize_finish()
  knots = bsobj._all_knots
  coefs = fit.params.values

  Xder1 = eval_bspline_basis_first_deriv(x, knots, 3)

  # Compute derivatives
  pred_deriv1 = np.dot(Xder1, coefs)

  return((x, pred_vals, pred_deriv1, Xder1))

def get_bsplines(x, df, degree, include_intercept, augment=[]):
  """
  Compute the B-spline and its first derivative for x

  Parameters
  ----------
  x : array-like
    Data to project into B-spline space
  df : 
    degrees of freedom. df - 1 basis vectors + intercept = df vectors
  degree : int
    Degree of B-spline
  include_intercept : bool
    Whether or not to include the intercept in the B-spline
  augment : array-like
    Values with which to augment x. Used to increase the range of the B-spline.
    These are removed when the projected X is returned. 

  Returns
  -------
  : B-splines vectors, B-spline vector derivatives

  """
  x = np.append(augment, x)
  X = dmatrix("bs(x, df={0}, degree={1}, include_intercept={2}) - 1"
              .format(df, degree, include_intercept), 
                        {"x": x}, return_type="dataframe").values
  bsobj = BS()
  bsobj.memorize_chunk(x, df=df, degree=degree, 
                          include_intercept=include_intercept)
  bsobj.memorize_finish()
  knots = bsobj._all_knots

  Xder = eval_bspline_basis_first_deriv(x, knots, degree)


  return((X[len(augment):, :], Xder[len(augment):, :]))


def predict_bspline(newx, oldx, df, degree=3, include_intercept=True, 
                      standardize=True, augment=[]):
  """
  Predict a new B-spline matrix and derivative from old x

  Parameters
  ----------
  newx : array-like
    New data on which to generate B-spline 
  oldx : array-like
    Original data on which B-spline was defined
  df : int
    Number of basis vectors

  Returns
  -------
  : (new B-spline design matrix, derivative of new B-spline design matrix)

  """

  oldx = np.append(augment, oldx)

  design_mat = dmatrix("bs(x, df={0}, degree={1}, include_intercept={2}) - 1"
                       .format(df, degree, include_intercept), {"x": oldx})
  Xnew = build_design_matrices([design_mat.design_info], {'x': newx})[0]

  # Get old knots
  bsobj = BS()
  bsobj.memorize_chunk(oldx, df=df, degree=degree, 
                             include_intercept=include_intercept)
  bsobj.memorize_finish()
  knots = bsobj._all_knots
  Xder_new = eval_bspline_basis_first_deriv(newx, knots, degree)

  # Standardize by 1 / exp(newx) if working in log(time)
  if standardize:
    tmat_new = (np.tile(np.exp(newx), Xder_new.shape[1])
                  .reshape(Xder_new.shape[1], len(newx)).T)
    Xder_new = Xder_new / tmat_new

  return((np.array(Xnew), np.array(Xder_new)))


def scale_X(X, time): 
  # Scale X by time
  newX = X / (np.tile(time, X.shape[1]).reshape(X.shape[1], len(time)).T)
  return(newX)

def eval_bspline_basis_first_deriv(x, knots, degree):
  # From patsy.splines. Use to calculate first derivative of basis function
  try:
      from scipy.interpolate import splev
  except ImportError: # pragma: no cover
      raise ImportError("spline functionality requires scipy")
  # 'knots' are assumed to be already pre-processed. E.g. usually you
  # want to include duplicate copies of boundary knots; you should do
  # that *before* calling this constructor.
  knots = np.atleast_1d(np.asarray(knots, dtype=float))
  assert knots.ndim == 1
  knots.sort()
  degree = int(degree)
  x = np.atleast_1d(x)
  if x.ndim == 2 and x.shape[1] == 1:
      x = x[:, 0]
  assert x.ndim == 1
  # XX FIXME: when points fall outside of the boundaries, splev and R seem
  # to handle them differently. I don't know why yet. So until we understand
  # this and decide what to do with it, I'm going to play it safe and
  # disallow such points.
  if np.min(x) < np.min(knots) or np.max(x) > np.max(knots):
      raise NotImplementedError("some data points fall outside the "
                                "outermost knots, and I'm not sure how "
                                "to handle them. (Patches accepted!)")
  # Thanks to Charles Harris for explaining splev. It's not well
  # documented, but basically it computes an arbitrary b-spline basis
  # given knots and degree on some specificed points (or derivatives
  # thereof, but we don't use that functionality), and then returns some
  # linear combination of these basis functions. To get out the basis
  # functions themselves, we use linear combinations like [1, 0, 0], [0,
  # 1, 0], [0, 0, 1].
  # NB: This probably makes it rather inefficient (though I haven't checked
  # to be sure -- maybe the fortran code actually skips computing the basis
  # function for coefficients that are zero).
  # Note: the order of a spline is the same as its degree + 1.
  # Note: there are (len(knots) - order) basis functions.
  n_bases = len(knots) - (degree + 1)
  basis = np.empty((x.shape[0], n_bases), dtype=float)
  for i in range(n_bases):
      coefs = np.zeros((n_bases,))
      coefs[i] = 1
      basis[:, i] = splev(x, (knots, coefs, degree), der=1)
  return basis


def extract_fit(fit, time, df, degree=3, include_intercept=True, samps=500,
                               numpoints=100, Zpoints=None):
  """
  Generated fitted predictions for seasonal infection risk

  Parameters
  ----------
  fit : PyStan object
    The fitted Stan model 
  time : array-like
    The time of infection/sampling data
  df : int
    The number of basis functions in the B-spline prediction
  degree : int
    The degree of the B-spline. Should be 3 for cubic
  include_intercept : bool
  samps : int
    The number of samples to draw from the estimated seasonal infection risk 
  numpoint : int
    The number of points at which to calculate seasonal infection risk using the
    model fits.  Points are taken uniformly between min(time) and max(time).
  Zpoints : array-like
    Vector of time-independent covariates that affect infection risk. If
    covariates were included in the model in a proportional hazard formulation.

  Returns
  -------
  : dict
    newtime : The time points over which the new predictions were made
    loghazard : The log hazard at each time point
    logsurvival : The log survival function at each time point
    logfdensity : the log density function at each time point
  """

  time_trans = (time / np.max(time)) + 0.5 # Re-scaling time for easier estimation

  # Predicting new basis vectors
  newtime = np.linspace(np.min(time_trans), np.max(time_trans), num=numpoints)

  Xnew, Xder_new = predict_bspline(np.log(newtime), np.log(time_trans), df)

  Xnew_noint = Xnew[:, 1:Xnew.shape[1]]
  Xder_new_noint = Xder_new[:, 1:Xder_new.shape[1]] 

  # Predictions. Draw samples
  all_betas = fit.extract('betas')['betas']
  all_intercepts = fit.extract('intercept')['intercept']

  # Are covariates included?
  if Zpoints is not None:
    all_alphas = fit.extract('alphas')['alphas']
    Z = np.tile(Zpoints, numpoints).reshape((numpoints, len(Zpoints)))

  N = all_betas.shape[0]
  draws = np.random.choice(np.arange(N), size=samps, replace=True)

  alls = np.empty((samps, Xnew_noint.shape[0]))
  allf = np.empty((samps, Xnew_noint.shape[0]))
  allh = np.empty((samps, Xnew_noint.shape[0]))

  for i, d in enumerate(draws):
    # Predict the survival and density functions
    betas = all_betas[d, :]
    intercept = all_intercepts[d]

    if Zpoints is not None:
      alphas = all_alphas[d, :]
      cov = np.dot(Z, alphas)
    else:
      cov = 0

    logcum = intercept + np.dot(Xnew_noint, betas) + cov
    logsurvival = -np.exp(logcum)
    loghazard = np.log(np.dot(Xder_new_noint, betas)) + logcum
    logfdensity = logsurvival + loghazard
    alls[i, :] = logsurvival
    allh[i, :] = loghazard
    allf[i, :] = logfdensity

  return(dict(loghazard=allh, logsurvival=alls, 
              logfdensity=allf,
              newtime=(newtime - 0.5)*np.max(time)))


def build_parametric_data(surv_df, recovered, leftcensored, theta, start, df, 
                          degree=3, include_intercept=True, 
                          cutoff=0.1719849357760231, calc_dob=False,
                          use_tau_end=False, rtol=0.01, yend=None, tau_min=None,
                          tau_end=None):
    """
    Format survival data for parametric analysis in Stan.
    
    Parameters
    ----------
    surv_df : DataFrame
        DataFrame with "time_in_season" (float), "infection" (bool), 
        and "sample_date" (DateTime) columns at least. 
        If "time_in_season" is not present, expects 
        "time_in_season_rising" and "time_in_season_falling". If "calc_dob=True",
        Also expected a column "dob" (DateTime).
        Finally, a column "serodata" can also be provided that contains the
        the the quantitative antibody samples. Only needed if rising and falling
        are being used, but not mandatory
    recovered : array-like
        The same len as surv_df. 1 means an individual is recovered, 0 otherwise
    left_censored : array-like
        1 means an individual is left-censored, 0 otherwise
    theta : dict
        Parameters of the antibody curve
    start : pandas._libs.tslibs.timestamps.Timestamp
        The date from which to start the estimation of seasonal infection risk
    df : int
        Degrees of freedom for B-spline
    degree : int
        Default to cubic B-spline (degree=3)
    include_intercept : bool
        Whether or not to include an intercept when fitting the B-spline
    cutoff : float
        The threshold for a seropositive. Default is np.log10(1 / 0.673)
    use_tau_end : bool
      If True, use tau_end instead of tau_falling when computing rising and
      falling probabilities. Also compute a new tau_falling from yend instead of 
      tau_rising. This is desirable if using fitting a left-censored
      rise and falling model.
    yend : float
      The antibody quantity below which a host is considered left-censored on the
      falling arm. Need to give it if use_tau_end is True. 
    calc_dob : bool
      Calculates the B-spline for a dob matrix. Expects the columns `dob` in
      surv_df
    tau_end : float
      Default is None. This allows tau_end to be directly passed.

    Returns
    -------
    : dict
      Keywords match all of the data parameters used in the Stan Models when
      fitting the survival analysis.
        
    """
    # Check for column. If not present, calculate both rising and falling
    compute_rf = "time_in_season" not in list(surv_df.columns)

    # Include both rising and falling estimates if desired
    if not compute_rf:
      time = surv_df.time_in_season.values
      maxtime = np.max(time)
      augment = []
    else:
      time = surv_df.time_in_season_falling.values
      maxtime = surv_df.time_in_season_rising.max()  # This should be from rising
      augment = [np.log((maxtime / maxtime) + 0.5)] # Augment max

    # Use DOB if desired. 
    if calc_dob:
      try:
        start_conv = start.to_datetime64()
      except:
        start_conv = start

      starts = pd.to_datetime(np.where(surv_df.dob.values < start_conv, 
                      start_conv, surv_df.dob.values))
      augment.append(np.log(0.5)) # augment below transformed min
      mintime = 0
    else:
      starts = np.repeat(start, len(surv_df))
      mintime = np.min(time)

    infected = surv_df.infection.astype(np.int).values
    sample_time = np.array([d.days for d in surv_df.sample_date - start])
    trans_time = (time / maxtime) + 0.5 # Transform time for easier estimation
    
    # Extract properties of the curve
    tau_rising, tau_falling, tau_max = get_abcurve_properties(theta, cutoff)
    if tau_end is None:
      tau_end = get_tau_end(theta, rtol=rtol)

    # If computing rising and falling for left-censoring use tau_end and set tau_rising = tau_lower = 0
    # If computing rising and falling probs for recovery, use tau_falling and tau_rising
    if use_tau_end:
      tau_upper = tau_end
      tau_lower = inverse_antibody_dynamics(yend, theta)[0]
    else:
      tau_upper = tau_falling
      tau_lower = tau_rising

    # Extract rising and falling probabilities.
    rfprobs = [compute_rising_falling_probs(tau_lower, 
                                                 tau_upper, 
                                                 tau_max, 
                                                 sd, 
                                                 st,
                                                 leftcensored=np.bool(lc),
                                                 tau_min=tau_min) 
                        for sd, st, lc in zip(surv_df.sample_date, starts, leftcensored)]
    prising, pfalling = list(zip(*rfprobs))

    # In some cases, p_falling is impossible based on the antibody level.
    # Can only determine this if serodata is a column that is given. 
    if "serodata" in list(surv_df.columns):

      rfdata = [inverse_antibody_dynamics(ab, theta) for ab in surv_df.serodata]
      rising, falling = list(zip(*rfdata))
      ind_rf = np.bitwise_and(np.array(rising) != -1, np.array(falling) == 500)
      prising = np.where(ind_rf, 1, prising)
      pfalling = np.where(ind_rf, 0, pfalling)

    # Compute B-splines
    X, Xder = get_bsplines(np.log(trans_time), df, degree, include_intercept, 
                            augment=augment)
    Xder = scale_X(Xder, trans_time) # Scale with the chain rule

    # Predict the dob B-spline if necessary
    if calc_dob:
      dob_time = np.array([d.days for d in surv_df.dob - start])
      trans_dob = (dob_time / maxtime) + 0.5
      trans_dob[trans_dob < 0.5] = 0.5
      Xdob, _ = predict_bspline(np.log(trans_dob), np.log(trans_time), df, augment=augment)
    else:
      Xdob = np.empty(X.shape)
    
    nus = (sample_time - tau_falling).astype(np.float)
    nus[nus < mintime] = mintime

    lower = time - tau_lower
    lower[lower < mintime] = mintime

    lc_time = sample_time - tau_end
    lc_time[lc_time < mintime] = mintime
    lc_time[lc_time > maxtime] = maxtime

    trans_nu = (nus / maxtime) + 0.5
    Xnu, _ = predict_bspline(np.log(trans_nu), np.log(trans_time), df, augment=augment)

    trans_lower = (lower / maxtime) + 0.5
    Xlower, _ = predict_bspline(np.log(trans_lower), np.log(trans_time), df, augment=augment)

    trans_lc = (lc_time / maxtime) + 0.5
    Xlc, _ = predict_bspline(np.log(trans_lc), np.log(trans_time), df, augment=augment)

    # Build the rising B-spline if necessary
    if compute_rf:
      rising_time = surv_df.time_in_season_rising.values
      trans_rising = (rising_time / maxtime) + 0.5
      Xrising, Xder_rising = predict_bspline(np.log(trans_rising), 
                                             np.log(trans_time), 
                                             df, augment=augment)
    else:
      Xrising = np.empty(X.shape)
      Xder_rising = np.empty(X.shape)

    # Build and return Stan data
    p = X.shape[1]
    standata = dict(N=X.shape[0], 
                    p=p - 1, 
                    X=X[:, 1:p], 
                    Xder=Xder[:, 1:p], 
                    infected=infected, 
                    recovered=recovered, 
                    leftcensored=leftcensored,
                    Xnu=Xnu[:, 1:p], 
                    Xlower=Xlower[:, 1:p],
                    Xlc=Xlc[:, 1:p],
                    Xrise=Xrising[:, 1:p],
                    Xder_rise=Xder_rising[:, 1:p],
                    prising=np.array(prising),
                    pfalling=np.array(pfalling),
                    Xdob=Xdob[:, 1:p])
    
    return(standata)




  