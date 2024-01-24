import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
import numpy as np
import common

class SafetyGymWrapper:

  def __init__(self, name, env, action_repeat=1, obs_key='state', act_key='action', size=(64, 64)):
    if "all" in name:
      self._dict_reward = True
      self._tasks = common.DOMAIN_TASK_IDS[name]
    self.name = name

    self._env = env
    self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_is_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._action_repeat = action_repeat
    self._size = size

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    if self._obs_is_dict:
      spaces = self._env.observation_space.spaces.copy()
    else:
      spaces = {self._obs_key: self._env.observation_space}
    return {
        **spaces,
        'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
    }

  @property
  def act_space(self):
    return {self._act_key: self._env.action_space}
  
  def eval_cases(self, task):
    return self._env.eval_cases

  def step(self, action):
    if not self._act_is_dict:
      action = action[self._act_key]

    if self._dict_reward:
        reward = []
    else:
        reward = 0.0
    for _ in range(self._action_repeat):
      obs, rew, done, info = self._env.step(action)
      if self._dict_reward:
        curr_reward = list(rew.values())
        if len(reward) == 0:
          reward = curr_reward
        else:
          reward = [sum(x) for x in zip(reward, curr_reward)]
      else:
          reward += rew or 0.0
      if done:
        break

    if not isinstance(obs, dict):
      obs = {self._obs_key: obs}
    obs['reward'] = reward
    obs['is_first'] = False
    obs['is_last'] = done
    obs['is_terminal'] = info.get('is_terminal', done)
    obs['image'] = self.render()

    if "task_completion" in info.keys():
      obs["task_completion"] = info["task_completion"]
    return obs
  
  def render(self):
    img = self._env.render(camera_id=0,
                          mode="rgb_array",
                          height=self._size[0],
                          width=self._size[1])
    return img

  def reset(self, env_params=None, task=None):
    obs, info = self._env.reset(env_params=env_params)
    if not isinstance(obs, dict):
      obs = {self._obs_key: obs}
    if self._dict_reward:
      reward = [0.0 for _ in self._tasks]
    else:
      reward = 0.0
    obs['reward'] = reward
    obs['is_first'] = True
    obs['is_last'] = False
    obs['is_terminal'] = False
    obs['image'] = self.render()
    if "task_completion" in info.keys():
      obs["task_completion"] = info["task_completion"]
    return obs
  
class CombinedEnvWrapper:

  def __init__(self, safety_gym_env, dmc_env, dmc_env_params=3, safety_gym_env_params=6):
    self.safety_gym_env = safety_gym_env
    self.dmc_env = dmc_env
    self.dmc_env_params = dmc_env_params
    self.safety_gym_env_params = safety_gym_env_params
    self.aug_env_params = max(self.dmc_env_params, self.safety_gym_env_params) + 1

    self.safety_gym_actions = int(self.safety_gym_env.act_space["action"].shape[0])
    self.dmc_actions = int(self.dmc_env.act_space["action"].shape[0])
    self.num_actions = max(self.safety_gym_actions, self.dmc_actions)

    self.dmc_tasks = self.dmc_env._tasks
    self.safety_gym_tasks = self.safety_gym_env._tasks
    self.name = self.dmc_env.name + "-" + self.safety_gym_env.name
    self.combined_tasks = common.DOMAIN_TASK_IDS[self.name]

    self.act_space = {"action": gym.spaces.Box(-1, 1, (self.num_actions,), dtype=np.float32)}
    self.obs_space = self.safety_gym_env.obs_space

    self.obs_keys = ["env_params", "image", "reward", "is_first", "is_last", "is_terminal", "task_completion"]
    self.domain_id_map = {"dmc": 0.0, "safety_gym": 1.0}

    self.dmc_eval_cases = dict()
    for key, cases in self.dmc_env.eval_cases(None).items():
      new_key = "dmc_" + key
      new_cases = [self.to_aug_env_params(case, self.domain_id_map["dmc"]) for case in cases]
      self.dmc_eval_cases[new_key] = new_cases

    self.safety_gym_eval_cases = dict()
    for key, cases in self.safety_gym_env.eval_cases(None).items():
      new_key = "safetygym_" + key
      new_cases = [self.to_aug_env_params(case, self.domain_id_map["safety_gym"]) for case in cases]
      self.safety_gym_eval_cases[new_key] = new_cases

  def eval_cases(self, task):
    if task in self.dmc_tasks:
      return self.dmc_eval_cases
    elif task in self.safety_gym_tasks:
      return self.safety_gym_eval_cases
    else:
      raise ValueError("Task not recognized.")

  def reset(self, env_params=None, task=None):
    # if the task is specified set domain appropriately
    if task is not None:
      if task in self.dmc_tasks:
        self.domain_id = self.domain_id_map["dmc"]
      elif task in self.safety_gym_tasks:
        self.domain_id = self.domain_id_map["safety_gym"]
      else:
        raise ValueError("Task not recognized.")

    # else if env params set use that to env the domain
    elif env_params is not None:
      self.domain_id = env_params[0]

    # otherwise choose randomly
    else:
      if np.random.uniform() < 0.5:
        self.domain_id = self.domain_id_map["dmc"]
      else:
        self.domain_id = self.domain_id_map["safety_gym"]

    # set the current env
    if np.isclose(self.domain_id, self.domain_id_map["safety_gym"]):
      self.current_env = self.safety_gym_env
    elif np.isclose(self.domain_id, self.domain_id_map["dmc"]):
      self.current_env = self.dmc_env
    else:
      raise ValueError("Domain ID not recognized.")

    obs = self.current_env.reset(env_params=self.to_original_env_params(env_params))

    if env_params is not None:
      self.current_env_params = env_params
    else:
      self.current_env_params = self.to_aug_env_params(obs["env_params"])
    
    if "task_completion" not in obs.keys():
      obs["task_completion"] = self.to_combined_task_completion()
    else:
      obs["task_completion"] = self.to_combined_task_completion(obs["task_completion"])

    obs["env_params"] = self.current_env_params.copy()
    if "state" in obs.keys():
      del obs["state"]
    
    obs["reward"] = np.zeros(len(self.combined_tasks))
    obs_fin = {key: obs[key] for key in self.obs_keys}
    return obs_fin
  
  def to_aug_env_params(self, original_env_params=None, domain_id=None):
    if original_env_params is None:
      return None
    original_params_size = len(original_env_params)
    new_env_params = np.zeros(self.aug_env_params)
    if domain_id is None:
      new_env_params[0] = self.domain_id
    else:
      new_env_params[0] = domain_id
    new_env_params[1:(1 + original_params_size)] = original_env_params
    return new_env_params
  
  def to_original_env_params(self, aug_env_params=None):
    if aug_env_params is None:
      return None
    original_env_params = aug_env_params[1:]
    if np.isclose(self.domain_id, self.domain_id_map["safety_gym"]):
      original_env_params = original_env_params[:self.safety_gym_env_params]
    else:
      original_env_params = original_env_params[:self.dmc_env_params]
    return original_env_params
  
  def step(self, action):
    action = action.copy()
    if self.current_env == self.safety_gym_env:
      action["action"] = action["action"][:self.safety_gym_actions]
    else:
      action["action"] = action["action"][:self.dmc_actions]

    obs = self.current_env.step(action)

    if "task_completion" not in obs.keys():
      obs["task_completion"] = self.to_combined_task_completion()
    else:
      obs["task_completion"] = self.to_combined_task_completion(obs["task_completion"])

    obs["env_params"] = self.current_env_params.copy()

    if "state" in obs.keys():
      del obs["state"]
    obs["reward"] = self.to_combined_reward(obs["reward"])
    obs_fin = {key: obs[key] for key in self.obs_keys}
    return obs_fin
  
  def to_combined_reward(self, env_reward):
    combined_reward = np.zeros(len(self.combined_tasks))

    for task in self.combined_tasks:
      if task in self.dmc_tasks and self.current_env == self.dmc_env:
        combined_reward[self.combined_tasks.index(task)] = env_reward[self.dmc_tasks.index(task)]
      elif task in self.safety_gym_tasks and self.current_env == self.safety_gym_env:
        combined_reward[self.combined_tasks.index(task)] = env_reward[self.safety_gym_tasks.index(task)]
    return combined_reward
  
  def to_combined_task_completion(self, task_completion=None):
    combined_task_completion = np.zeros(len(self.combined_tasks))
    if task_completion is None:
      return combined_task_completion
    
    for task in self.combined_tasks:
      if task in self.safety_gym_tasks and self.current_env == self.safety_gym_env:
        combined_task_completion[self.combined_tasks.index(task)] = task_completion[self.safety_gym_tasks.index(task)]
    return combined_task_completion

class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if task == 'all':
        self._dict_reward = True
        self._tasks = common.DOMAIN_TASK_IDS[name]
    else:
        self._dict_reward = False
    self.name = name
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if domain == 'manip':
      from dm_control import manipulation
      self._env = manipulation.load(task + '_vision')
    elif domain == 'locom':
      from dm_control.locomotion.examples import basic_rodent_2020
      self._env = getattr(basic_rodent_2020, task)()
    else:
      from envs.dm_control import suite
      self._env = suite.load(domain, task)
    self._action_repeat = action_repeat
    self._size = size
    if camera in (-1, None):
      camera = dict(
          quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
          quadruped_fetch=2, locom_rodent_maze_forage=1,
          locom_rodent_two_touch=1,
      ).get(name, 0)
    self._camera = camera
    self._ignored_keys = []
    for key, value in self._env.observation_spec().items():
      if value.shape == (0,):
        print(f"Ignoring empty observation key '{key}'.")
        self._ignored_keys.append(key)

  @property
  def obs_space(self):
    spaces = {
        'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
    }
    for key, value in self._env.observation_spec().items():
      if key in self._ignored_keys:
        continue
      if value.dtype == np.float64:
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
      elif value.dtype == np.uint8:
        spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
      else:
        raise NotImplementedError(value.dtype)
    return spaces
  
  def eval_cases(self, task):
    return self._env.task.eval_cases

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return {'action': action}

  def step(self, action):
    assert np.isfinite(action['action']).all(), action['action']
    if self._dict_reward:
        reward = []
    else:
        reward = 0.0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action['action'])
      if self._dict_reward:
        curr_reward = list(time_step.reward.values())
        if len(reward) == 0:
          reward = curr_reward
        else:
          reward = [sum(x) for x in zip(reward, curr_reward)]
      else:
          reward += time_step.reward or 0.0
      if time_step.last():
        break
    assert time_step.discount in (0, 1)
    obs = {
        'reward': reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs

  def reset(self, env_params=None, task=None):
    time_step = self._env.reset(env_params)
    if self._dict_reward:
      reward = [0.0 for _ in self._tasks]
    else:
      reward = 0.0
    obs = {
        'reward': reward,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs

class Dummy:

  def __init__(self):
    pass

  @property
  def obs_space(self):
    return {
        'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
    }

  @property
  def act_space(self):
    return {'action': gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

  def step(self, action):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': False,
        'is_last': False,
        'is_terminal': False,
    }

  def reset(self):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
    return obs

  def reset(self, env_params=None, task=None):
    self._step = 0
    return self._env.reset(env_params=env_params, task=task)


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.act_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class OneHotAction:

  def __init__(self, env, key='action'):
    assert hasattr(env.act_space[key], 'n')
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    shape = (self._env.act_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index})

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.act_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class ResizeImage:

  def __init__(self, env, size=(64, 64)):
    self._env = env
    self._size = size
    self._keys = [
        k for k, v in env.obs_space.items()
        if len(v.shape) > 1 and v.shape[:2] != size]
    print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
    if self._keys:
      from PIL import Image
      self._Image = Image

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
    return spaces

  def step(self, action):
    obs = self._env.step(action)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def reset(self):
    obs = self._env.reset()
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    image = self._Image.fromarray(image)
    image = image.resize(self._size, self._Image.NEAREST)
    image = np.array(image)
    return image


class RenderImage:

  def __init__(self, env, key='image'):
    self._env = env
    self._key = key
    self._shape = self._env.render().shape

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
    return spaces

  def step(self, action):
    obs = self._env.step(action)
    obs[self._key] = self._env.render('rgb_array')
    return obs

  def reset(self):
    obs = self._env.reset()
    obs[self._key] = self._env.render('rgb_array')
    return obs


class Async:

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _CLOSE = 4
  _EXCEPTION = 5

  def __init__(self, constructor, strategy='thread'):
    self._pickled_ctor = cloudpickle.dumps(constructor)
    if strategy == 'process':
      import multiprocessing as mp
      context = mp.get_context('spawn')
    elif strategy == 'thread':
      import multiprocessing.dummy as context
    else:
      raise NotImplementedError(strategy)
    self._strategy = strategy
    self._conn, conn = context.Pipe()
    self._process = context.Process(target=self._worker, args=(conn,))
    atexit.register(self.close)
    self._process.start()
    self._receive()  # Ready.
    self._obs_space = None
    self._act_space = None

  def access(self, name):
    self._conn.send((self._ACCESS, name))
    return self._receive

  def call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      pass  # The connection was already closed.
    self._process.join(5)

  @property
  def obs_space(self):
    if not self._obs_space:
      self._obs_space = self.access('obs_space')()
    return self._obs_space

  @property
  def act_space(self):
    if not self._act_space:
      self._act_space = self.access('act_space')()
    return self._act_space

  def step(self, action, blocking=False):
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=False):
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to environment worker.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, conn):
    try:
      ctor = cloudpickle.loads(self._pickled_ctor)
      env = ctor()
      conn.send((self._RESULT, None))  # Ready.
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print('Error in environment process: {}'.format(stacktrace))
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      try:
        conn.close()
      except IOError:
        pass  # The connection was already closed.
