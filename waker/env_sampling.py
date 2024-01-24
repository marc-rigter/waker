import numpy as np
import tensorflow as tf
import collections
import common


class EnvSampler(object):
  """ Base class for setting the environment parameters.

  Args:
    interval: number of steps between which to estimate the change in uncertainty.
    gamma_uncert_reduct: the discount factor for smoothing the change in uncertainty
  """
  def __init__(self, task, interval=10000, gamma_uncert_reduct=0.95):
    self.interval = interval
    self.gamma_uncert_reduct = gamma_uncert_reduct
    self.gamma_uncert = 1 - 1 / interval # for smoothing the uncertainty estimate
    self.episode_num = 0
    self.task = task

    # initialise buffer of environment params and uncertainty estimates
    self.ens_uncert = collections.OrderedDict()
    self.episodes = collections.OrderedDict()
    self.prev_ens_uncert = collections.OrderedDict()
    self.ens_uncert_change = collections.OrderedDict()
    self.initialised = False
  
  def get_env_params(self):
    """ Set the environment parameters.
    """
    self.episode_num += 1
    env_params = self.sample_env_params()

    # record number of times context has been selected
    if env_params is not None:
      dict_key = common.get_dict_key(env_params)
      self.episodes[dict_key] = 1 + self.episodes.get(dict_key, 0)
    return env_params
  
  def sample_env_params(self):
    """ Sample the environment parameters according to sampling strategy.
    """
    raise NotImplementedError
  
  def sample_env_params_dr(self):
    """ Sample the environment parameters according to the domain randomisation distribution 
    over the environment parameters.
    """
    if "terrain" in self.task and "clean" not in self.task:
      amplitude_range = [0.0, 1.0]
      length_scale_range = [0.2, 2.0]

      # discretise to improve logging
      ampl_list = np.linspace(*amplitude_range, num=int((amplitude_range[1] - amplitude_range[0])/0.05 + 1))
      length_list = np.linspace(*length_scale_range, num=int((length_scale_range[1] - length_scale_range[0])/0.05 + 1))

      # sample uniform distribution over amplitude and length scale
      amplitude = np.random.choice(ampl_list)
      length_scale = np.random.choice(length_list)
      interp = 0 # cosine interpolation
      params = np.array([amplitude, length_scale, interp])
      return params
    
    elif "clean" in self.task and "terrain" not in self.task:

      # sample arena size
      arena_size = np.random.choice([0, 1, 2, 3, 4])
      max_object_num = arena_size

      # sample number of blocks
      object_num = np.random.randint(low=0, high=max_object_num + 1)

      # sample color of blocks
      object_1_num = np.random.randint(low=0, high=object_num + 1)
      object_2_num = object_num - object_1_num
      params = np.array([arena_size, object_1_num, object_2_num, 0, 0, 0])
      return params
    
    else:
      return None

  def update_ensemble_uncert(self, data, expl_seq):
    """ Update the uncertainty estimates of the ensemble using imagination rollouts
    and the associated intrinsic rewards.

    Args: 
      data: dictionary of data from the replay buffer used to generate imagination rollouts.
      expl_seq: dictionary of imagination rollout data.
    """

    # compute average intrinsic reward in sequence
    env_params = data["env_params"].reshape([-1] + list(data["env_params"].shape[2:])).numpy()
    intr_rew = tf.reduce_mean(expl_seq["reward"], axis=0).numpy()

    # update ensemble uncertainty estimates via smoothed average
    for i in range(env_params.shape[0]):
      dict_key = common.get_dict_key(env_params[i, :])
      if dict_key in self.ens_uncert.keys():
        self.ens_uncert[dict_key] = self.gamma_uncert * self.ens_uncert[dict_key] + \
           (1-self.gamma_uncert) * intr_rew[i]
      else:
        self.ens_uncert[dict_key] = intr_rew[i]
    self.initialised = True
    return 
  
  def update_ensemble_uncert_changes(self, step):
    """ Update running average of the change in ensemble uncertainty estimates.
    """

    if not self.initialised:
      return
    
    # give estimates time to stabilise
    if step < 5 * self.interval:
      return 
    
    # only update every interval steps
    if not step % self.interval == 0:
      return
    
    # compute change in uncertainty estimate
    for key in self.prev_ens_uncert.keys():
      change = self.prev_ens_uncert[key] - self.ens_uncert[key] 
      if key in self.ens_uncert_change.keys():
        self.ens_uncert_change[key] = self.gamma_uncert_reduct * self.ens_uncert_change[key] + \
          (1 - self.gamma_uncert_reduct) * change
      else:
        self.ens_uncert_change[key] = (1 - self.gamma_uncert_reduct) * change

    # update previous uncertainty estimate
    self.prev_ens_uncert = self.ens_uncert.copy()

  def get_metrics(self):
    """ Get metrics for logging.
    """
    if not self.ens_uncert:
      return dict()
    
    # metrics relating to ensemble uncertainty estimates
    metrics = dict()
    param_keys = sorted(self.ens_uncert.keys())
    for key in param_keys:
      if key in self.episodes.keys():
        num_eps = self.episodes.get(key)
        prop_eps = self.episodes.get(key) / sum([eps for keys, eps in self.episodes.items()])
      else:
        num_eps, prop_eps = 0, 0
      metrics["num_episodes/" + str(key)] = num_eps
      metrics["ens_uncert_magnitude/" + str(key)] = self.ens_uncert[key]
      metrics["proportion_episodes/" + str(key)] = prop_eps

    # metrics for change in uncertainty estimates
    param_keys = sorted(self.ens_uncert_change.keys())
    for key in param_keys:
      metrics["ens_uncert_reduction/" + str(key)] = self.ens_uncert_change[key]
    return metrics
  
class Random(EnvSampler):
  """ Set the environment parameters according to the domain randomisation distribution.
  """

  def __init__(self, task,
      **kwargs):
    super().__init__(task=task)
  
  def sample_env_params(self):
    return self.sample_env_params_dr()
  
class HardestEnvOracle(EnvSampler):
  """ Set the environment to the hardest configuration using expert knowledge
  """

  def __init__(self, task):
    super().__init__(task=task)
  
  def sample_env_params(self):
    if "terrain" in self.task and "clean" not in self.task:
      # highest amplitude shortest length scale
      return np.array([1.0, 0.2, 0])
    elif "clean" in self.task and "terrain" not in self.task:
      # maximum blocks of mixed colour
      return np.array([4, 2, 2, 0, 0, 0])
    else:
      raise NotImplementedError
  
class ReweightingOracle(EnvSampler):
  """ Reweight the environment parameters using expert domain knowledge
  """

  def __init__(self, task):
    super().__init__(task=task)
  
  def sample_env_params(self):
    if np.random.uniform() < 0.8:
      if "terrain" in self.task and "clean" not in self.task:
        # high amplitude and short length scale
        return np.array([np.random.choice([0.8, 0.85, 0.9, 0.95, 1.0]), np.random.choice([0.2, 0.25, 0.3, 0.35, 0.4]), 0])
      elif "clean" in self.task and "terrain" not in self.task:
        # maximum blocks
        green_blocks = np.random.randint(low=0, high=5)
        return np.array([4, green_blocks, 4 - green_blocks, 0, 0, 0])
      else:
        raise NotImplementedError
      
    # otherwise use DR distribution
    else:
      return self.sample_env_params_dr()
    
class GradualExpansion(EnvSampler):
  """ Gradually expand the environment parameters from a single environment to all environments.
  """

  def __init__(self, task):
    super().__init__(task=task)
  
  def sample_env_params(self):
    if len(self.ens_uncert.keys()) == 0:
      return self.sample_env_params_dr()
    
    # sample from DR distribution of parameters seen so far
    if np.random.uniform() < 0.8:
      params_visited = list(self.episodes.keys())
      while True:
        sampled_params = self.sample_env_params_dr()
        sampled_params_key = common.get_dict_key(sampled_params)
        if sampled_params_key in params_visited:
          print("Sampled from DR distribution of previously visited environments.")
          return sampled_params

    # otherwise use DR distribution
    else:
      return self.sample_env_params_dr()

class ParamList(object):
  """ Sequentially set the environment parameters from a list of parameters."""
  
  def __init__(self, param_list, name):
    self._param_list = param_list
    self._param_idx = 0
    self._name = name

  def get_env_params(self):
    env_params = self._param_list[self._param_idx]

    self._param_idx += 1
    if self._param_idx >= len(self._param_list):
      self._param_idx = 0
    print(f"Parameter list {self._name}, setting params to: {env_params}")
    return env_params

def convert(value):
  value = np.array(value)
  if np.issubdtype(value.dtype, np.floating):
    return value.astype(np.float32)
  elif np.issubdtype(value.dtype, np.signedinteger):
    return value.astype(np.int32)
  elif np.issubdtype(value.dtype, np.uint8):
    return value.astype(np.uint8)
  return value
