import numpy as np
from env_sampling import EnvSampler

class WAKER(EnvSampler):

  def __init__(
      self, 
      task,
      sample_uncert="ensemble_magnitude",
      temp=1.0,
      random_prob=0.2,
      interval=10000,
      gamma_uncert_reduct=0.95,
      combined=False
    ):
    """ Class to sample environment contexts according to ensemble-based
    uncertainty estimate.

    Args:
      sample_uncert: the type of error estimate to use for sampling.
      temp: temperature of Boltzmann distribution for sampling envs.
      random_prob: probability of sampling a random context for exploration.
      interval: number of steps between updates of change in uncertainty.
      gamma_uncert_reduct: for smoothing the change in uncertainty estimate.
      combined: whether we are estimating uncertainty over multiple different domains
    """
    self.sample_uncert = sample_uncert
    self.random_prob = random_prob
    self.temp = temp
    self.combined = combined
    super().__init__(task=task, interval=interval, gamma_uncert_reduct=gamma_uncert_reduct)
  
  def get_metrics(self):
    """ Return metrics for sampling distribution. """
    if not self.ens_uncert:
      return dict()
    
    metrics = super().get_metrics()

    dist, param_keys, uncert_normed = self.get_sampling_distribution()
    if dist is None:
      return metrics
    
    param_keys, dist, uncert_normed = zip(*sorted(zip(param_keys, dist, uncert_normed)))
    for i in range(len(param_keys)):
      metrics[f"{self.sample_uncert}_sampling_prob/" + str(param_keys[i])] = dist[i]
      metrics[f"{self.sample_uncert}_uncert_normed/" + str(param_keys[i])] = uncert_normed[i]
    return metrics

  def get_sampling_distribution(self):
    """ Get the sampling distribution according to uncertainty estimate.
    """

    if self.sample_uncert == "ensemble_magnitude":
      return self.sampling_distribution(self.ens_uncert)
    elif self.sample_uncert == "ensemble_reduction":
      return self.sampling_distribution(self.ens_uncert_change, norm="mean")
    else:
      raise NotImplementedError
  
  def sampling_distribution(self, uncert_est, norm="mean_std"):
    """ Sampling distribution is a softmax over normalised ensemble uncertainties."""

    if len(uncert_est.keys()) == 0:
      return None, None, None

    param_keys = list(uncert_est)
    uncertainties = np.array(list(uncert_est.values()))

    if self.combined:
      param_keys, uncertainties_normed = self.norm_by_domain(param_keys, uncertainties, norm)
    else:
      param_keys, uncertainties_normed = self.norm(param_keys, uncertainties, norm)
    
    exp = np.exp(uncertainties_normed / self.temp)
    dist = exp / np.sum(exp)
    return dist, param_keys, uncertainties_normed
  
  def norm(self, param_keys, uncertainties, norm="mean_std"):
    """ Normalise uncertainty estimates """

    if norm == "mean_std":
      uncertainties_normed = (uncertainties - np.mean(uncertainties)) \
        / (np.std(uncertainties) + 1e-6)
    elif norm == "mean":
      uncertainties_normed = uncertainties / np.mean(np.abs(uncertainties))
    elif norm == "none":
      uncertainties_normed = uncertainties
    else:
      raise NotImplementedError
    return param_keys, uncertainties_normed
  
  def norm_by_domain(self, param_keys, uncertainties, norm="mean_std"):
    """ Normalise uncertainty estimates in each domain separately if training world 
    model for multiple domains """

    domain_params = [key.split(" ")[0] for key in param_keys]
    domain_params = list(np.unique(domain_params))

    # loop through each domain
    all_keys_temp = []
    all_norm_uncert_temp = []
    for dom_par in domain_params:
      domain_param_keys = []
      domain_uncert = []

      # find those keys that correspond to this domain
      for key, uncert in zip(param_keys, uncertainties):
        if dom_par in key.split(" ")[0]:
          domain_param_keys.append(key)
          domain_uncert.append(uncert)

      # normalize the uncertainties for this domain
      domain_uncert = np.array(domain_uncert)
      _, domain_uncert_norm = self.norm(domain_param_keys, domain_uncert, norm)

      # store all normalized uncertainties
      all_keys_temp.append(domain_param_keys)
      all_norm_uncert_temp.append(domain_uncert_norm.tolist())

    # make the lists the same length to prevent bias towards domains with more
    # params
    max_len = max([len(keys) for keys in all_keys_temp])
    all_keys = []
    all_norm_uncert  = []
    for i in range(len(all_keys_temp)):
      domain_param_keys = all_keys_temp[i]
      domain_uncert_norm = all_norm_uncert_temp[i]

      new_domain_param_keys = domain_param_keys
      new_domain_uncert_norm = domain_uncert_norm
      if len(new_domain_param_keys) < max_len:
        for _ in range(max_len - len(domain_param_keys)):
          ind = np.random.randint(low=0, high=len(domain_param_keys))
          new_domain_param_keys.append(domain_param_keys[ind])
          new_domain_uncert_norm.append(domain_uncert_norm[ind])

      all_keys.extend(new_domain_param_keys)
      all_norm_uncert.extend(new_domain_uncert_norm)

    all_norm_uncert = np.array(all_norm_uncert)
    return all_keys, all_norm_uncert
    
  def sample_env_params(self):
    """ Sample environment according to Boltzmann distribution.
    """

    # if not initialised, sample randomly
    dist, param_keys, _ = self.get_sampling_distribution()
    if dist is None:
      print(f"Uncertainty estimates not yet initialised, sampling random env params.")
      return self.sample_env_params_dr()

    # with some probability randomly sample new context
    if np.random.uniform() < self.random_prob:
      print(f"Random sample: sampling environment randomly.")
      return self.sample_env_params_dr()
    
    sample_id = np.random.choice(len(dist), p=dist)
    env_params = np.fromstring(param_keys[sample_id][1:-1], sep=" ")
    print(f"Sampling from Boltzmann distribution for {self.sample_uncert} uncertainty, env parameters: {env_params}")
    return env_params

