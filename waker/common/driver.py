import numpy as np


class Driver:

  def __init__(self, envs, **kwargs):
    self._envs = envs
    self._kwargs = kwargs
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._on_calls = []
    self._act_spaces = [env.act_space for env in envs]
    self._act_is_discrete = [hasattr(s['action'], 'n') for s in self._act_spaces]
    self.total_episodes = 0
    self.reset()

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def on_call(self, callback):
    self._on_calls.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def __call__(self, policy, steps=0, episodes=0, env_sampler=None, task=None):
    step, episode = 0, 0
    eps = []
    while step < steps or episode < episodes:
      if env_sampler is None:
        obs = {
            i: self._envs[i].reset(env_sampler, task=task)
            for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
      else:
        obs = {
            i: self._envs[i].reset(env_params=env_sampler.get_env_params(), task=task)
            for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
        
      for i, ob in obs.items():
        self._obs[i] = ob() if callable(ob) else ob
        if self._act_is_discrete[i]:
          act = {k: np.zeros(v.n) for k, v in self._act_spaces[i].items()}
        else:
          act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
        self._eps[i] = [tran]
      obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
      actions, self._state = policy(obs, self._state, **self._kwargs)
      actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
      assert len(actions) == len(self._envs)
      obs = [e.step(a) for e, a in zip(self._envs, actions)]
      obs = [ob() if callable(ob) else ob for ob in obs]
      for i, (act, ob) in enumerate(zip(actions, obs)):
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
        self._eps[i].append(tran)
        step += 1
        if ob['is_last']:
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          [fn(ep, self.total_episodes, **self._kwargs) for fn in self._on_episodes]
          eps.append(ep)
          episode += 1
          self.total_episodes += 1
      self._obs = obs
    [fn(eps, **self._kwargs) for fn in self._on_calls]
    

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value
