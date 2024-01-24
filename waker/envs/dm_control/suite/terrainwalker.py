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

"""Planar Walker Domain."""

import collections
import numpy as np

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

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8
_SPIN_SPEED = 5
_HEIGHTFIELD_ID = 0


SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('terrainwalker.xml'), common.ASSETS


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Stand task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = TerrainWalker(move_speed=0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = TerrainWalker(move_speed=_WALK_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = TerrainWalker(move_speed=_RUN_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add('benchmarking')
def all(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns task using all reward functions."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = TerrainWalker(all=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Walker domain."""

  def torso_upright(self):
    """Returns projection from z-axes of torso to the z-axes of world."""
    return self.named.data.xmat['torso', 'zz']

  def torso_height(self):
    """Returns the height of the torso relative to foot."""
    return self.named.data.xpos['torso', 'z'] - self.named.data.xpos['right_foot', 'z']

  def horizontal_velocity(self):
    """Returns the horizontal velocity of the center-of-mass."""
    return self.named.data.sensordata['torso_subtreelinvel'][0]

  def orientations(self):
    """Returns planar orientations of all bodies."""
    return self.named.data.xmat[1:, ['xx', 'xz']].ravel()
  
  def angmomentum(self):
      """Returns the angular momentum of torso of about Y axis."""
      return self.named.data.subtree_angmom['torso'][1]

class TerrainWalker(base.Task):
  """A planar walker task with randomised terrain."""

  def __init__(self, move_speed=0.0, all=False, random=None, sample_list=True):
    """Initializes an instance of `TerrainWalker`.

    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._move_speed = move_speed
    self._amplitude_range = [0.0, 1.0]
    self._length_scale_range = [0.2, 2.0]
    self._interp_range = [0, 1, 2]
    self._sample_list = sample_list
    self._all = all

    eval_cases = {
      "OOD_stairs_large": [np.array([-1, 0, 0]) for _ in range(20)],
      "OOD_steep": [np.array([0.8, 0.15, 0]) for _ in range(20)],
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
    """Sets the state of the environment at the start of each episode.

    In 'standing' mode, use initial orientation and small velocities.
    In 'random' mode, randomize joint angles and let fall to the floor.

    Args:
      physics: An instance of `Physics`.

    """

    if params is None:
      params = self.get_random_params()
    self.env_params = params

    nrow = physics.model.hfield_nrow[0]
    ncol = physics.model.hfield_ncol[0]
    if params[0] < -0.5:
      # OOD case
      height_profile = generate_step_profile(ncol, physics.model.hfield_size[_HEIGHTFIELD_ID][0])
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

  def get_observation(self, physics):
    """Returns an observation of body orientations, height and velocites."""
    obs = collections.OrderedDict()
    obs['orientations'] = physics.orientations()
    obs['height'] = physics.torso_height()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    standing = rewards.tolerance(physics.torso_height(),
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT/2)
    upright = (1 + physics.torso_upright()) / 2
    stand_reward = (3*standing + upright) / 4

    # if all tasks return reward for all tasks
    if self._all:
      walk_reward = rewards.tolerance(
                1 * physics.horizontal_velocity(),
                bounds=(_WALK_SPEED, float('inf')),
                margin=_WALK_SPEED / 2,
                value_at_margin=0.5,
                sigmoid='linear')
      
      walk_bwd_reward = rewards.tolerance(
                -1 * physics.horizontal_velocity(),
                bounds=(_WALK_SPEED, float('inf')),
                margin=_WALK_SPEED / 2,
                value_at_margin=0.5,
                sigmoid='linear')

      run_reward = rewards.tolerance(
          1 * physics.horizontal_velocity(),
          bounds=(_RUN_SPEED, float('inf')),
          margin=_RUN_SPEED / 2,
          value_at_margin=0.5,
          sigmoid='linear')

      flip_reward = rewards.tolerance(1 *
                                      physics.angmomentum(),
                                      bounds=(_SPIN_SPEED, float('inf')),
                                      margin=_SPIN_SPEED,
                                      value_at_margin=0,
                                      sigmoid='linear')

      reward_dict = {
          'stand': stand_reward,
          'walk': stand_reward * (5*walk_reward + 1) / 6,
          'run': stand_reward * (5*run_reward + 1) / 6,
          'flip': flip_reward,
          'walk-bwd': stand_reward * (5*walk_bwd_reward + 1) / 6,
      }
      return reward_dict
    
    # otherwise return reward according to move speed
    else:
      move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                      bounds=(self._move_speed, float('inf')),
                                      margin=self._move_speed/2,
                                      value_at_margin=0.5,
                                      sigmoid='linear')
      return stand_reward * (5*move_reward + 1) / 6