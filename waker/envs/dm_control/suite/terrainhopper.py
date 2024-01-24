# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Hopper domain."""

import collections

from dm_control import mujoco
from envs.dm_control.rl import control
from envs.dm_control.suite import base
from envs.dm_control.suite import common
from envs.dm_control.suite.utils import randomizers
from envs.dm_control.suite.utils.terrain import Interp, PerlinNoise, generate_step_profile
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.mujoco.wrapper import mjbindings
mjlib = mjbindings.mjlib
import numpy as np


SUITE = containers.TaggedTasks()

_CONTROL_TIMESTEP = .02  # (Seconds)

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2
_HEIGHTFIELD_ID = 0

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('terrainhopper.xml'), common.ASSETS


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns a Hopper that strives to stand upright, balancing its pose."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = TerrainHopper(hopping=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def hop(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns a Hopper that strives to hop forward."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = TerrainHopper(hopping=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add('benchmarking')
def all(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns a Hopper that receives rewards for multiple tasks."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = TerrainHopper(all=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)



class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Hopper domain."""

  def height(self):
    """Returns height of torso with respect to foot."""
    return (self.named.data.xipos['torso', 'z'] -
            self.named.data.xipos['foot', 'z'])

  def speed(self):
    """Returns horizontal speed of the Hopper."""
    return self.named.data.sensordata['torso_subtreelinvel'][0]

  def touch(self):
    """Returns the signals from two foot touch sensors."""
    return np.log1p(self.named.data.sensordata[['touch_toe', 'touch_heel']])


class TerrainHopper(base.Task):
  """A Hopper's `Task` to train a standing and a jumping Hopper."""

  def __init__(self, hopping=False, all=False, random=None, sample_list=True):
    """Initialize an instance of `TerrainHopper`.

    Args:
      hopping: Boolean, if True the task is to hop forwards, otherwise it is to
        balance upright.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._hopping = hopping
    self._all = all
    self._amplitude_range = [0.0, 1.0]
    self._length_scale_range = [0.15, 1.5]
    self._sample_list = sample_list

    eval_cases = {
      "OOD_stairs_large": [np.array([-1, 0, 0]) for _ in range(20)],
      "OOD_steep": [np.array([0.8, 0.1125, 0]) for _ in range(20)],
      "OOD_max": [np.array([1.0, 0.15, 0]) for _ in range(20)],
    }
    super().__init__(random=random, eval_cases=eval_cases)

  def get_random_params(self):
    """ Sample parameters uniformly at random"""
    if self._sample_list:
      ampl_list = np.linspace(*self._amplitude_range, num=int((self._amplitude_range[1] - self._amplitude_range[0])/0.05 + 1))
      length_list = np.linspace(*self._length_scale_range, num=int((self._length_scale_range[1] - self._length_scale_range[0])/0.05 + 1))
      amplitude = np.random.choice(ampl_list)
      length_scale = np.random.choice(length_list)
    else:
      amplitude = np.random.uniform(*self._amplitude_range)
      length_scale = np.random.uniform(*self._length_scale_range)
    interp = 0 # cosine interpolation
    params = np.array([amplitude, length_scale, interp])
    return params

  def initialize_episode(self, physics, params=None):
    """Sets the state of the environment at the start of each episode."""
    if params is None:
      params = self.get_random_params()
    self.env_params = params

    nrow = physics.model.hfield_nrow[0]
    ncol = physics.model.hfield_ncol[0]
    if params[0] < -0.5:
      # OOD case
      height_profile = generate_step_profile(ncol, physics.model.hfield_size[_HEIGHTFIELD_ID][0], width=0.3)
    else:
      amplitude = params[0]
      length_scale = params[1]
      interp = Interp(int(params[2])).name
      freq = 1 / length_scale
      x = np.linspace(0, physics.model.hfield_size[_HEIGHTFIELD_ID][0], num=ncol)

      noise = PerlinNoise(
        amplitude=amplitude,
        frequency=freq,
        interp=interp)
      height_profile = np.array([noise.get(x[i]) for i in range(x.size)])

      # Normalize the height profile to be between 0 and 1.
      height_profile = np.expand_dims(height_profile, 1) / (self._amplitude_range[1] - self._amplitude_range[0]) / 2 + 0.5
    height_data = np.repeat(height_profile, nrow, axis=1)
    height_data = np.swapaxes(height_data, 0, 1)
    start_idx = physics.model.hfield_adr[_HEIGHTFIELD_ID]
    physics.model.hfield_data[start_idx:start_idx+nrow*ncol] = height_data.ravel()
    super().initialize_episode(physics)

    # If we have a rendering context, we need to re-upload the modified
    # heightfield data.
    if physics.contexts:
      with physics.contexts.gl.make_current() as ctx:
        ctx.call(mjlib.mjr_uploadHField,
                 physics.model.ptr,
                 physics.contexts.mujoco.ptr,
                 _HEIGHTFIELD_ID)
     
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    self._timeout_progress = 0
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of positions, velocities and touch sensors."""
    obs = collections.OrderedDict()
    # Ignores horizontal position to maintain translational invariance:
    obs['position'] = physics.data.qpos[1:].copy()
    obs['velocity'] = physics.velocity()
    obs['touch'] = physics.touch()
    return obs

  def get_reward(self, physics):
    """Returns a reward applicable to the performed task."""
    standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
    hopping = rewards.tolerance(physics.speed(),
                                bounds=(_HOP_SPEED, float('inf')),
                                margin=_HOP_SPEED/2,
                                value_at_margin=0.5,
                                sigmoid='linear')
    small_control = rewards.tolerance(physics.control(),
                                        margin=1, value_at_margin=0,
                                        sigmoid='quadratic').mean()
    
    backward_hopping = rewards.tolerance(-1 * physics.speed(),
                                bounds=(_HOP_SPEED, float('inf')),
                                margin=_HOP_SPEED/2,
                                value_at_margin=0.5,
                                sigmoid='linear')
    
    if self._all:
      rews = {
        'hop': standing * hopping, 
        'stand': standing * small_control,
        'hop-bwd': standing * backward_hopping, 
      }
      return rews
    elif self._hopping:
      return standing * hopping
    else:
      small_control = (small_control + 4) / 5
      return standing * small_control