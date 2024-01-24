#!/usr/bin/env python

import gym
import gym.spaces
import numpy as np
import common
from PIL import Image
from copy import deepcopy
from collections import OrderedDict
import mujoco_py
from mujoco_py import MjViewer, MujocoException, const, MjRenderContextOffscreen

from envs.safety_gym.envs.engine import Engine

import sys


class ReplayEngine(Engine):

    '''
    ReplayEngine: a class to enable resetting environments to earlier configurations.

    '''
    def __init__(self, config={}, param_list=None, eval_cases=None):
        super().__init__(config=config)
        self._param_list = param_list
        self.eval_cases = eval_cases

    @property
    def max_num_objects(self):
        return self.buttons_num + self.hazards_num + self.vases_num + self.pillars_num + self.gremlins_num
    
    def env_params_from_layout(self, layout):
        ''' Stores the number of each type of object in a numpy array'''
        
        num_config_params = len(self.configurable_types)
        env_params = np.zeros(num_config_params + 1)
        env_params[0] = float(self.arena)
        for i, object_type in enumerate(self.configurable_types):
            obj_num = self.num_objects_in_layout(object_type, layout)
            env_params[i + 1] = float(obj_num)
        return env_params
    
    def env_params_to_obj_dict(self, env_params):
        num_config_params = len(self.configurable_types)
        obj_dict = dict()
        obj_dict["arena"] = int(env_params[0])
        for i in range(num_config_params):
            param_name = self.configurable_types[i]
            obj_dict[param_name] = int(env_params[i + 1])
        return obj_dict

    def reset(self, env_params=None):
        ''' Reset the physics simulation and return observation '''
        self._seed += 1  # Increment seed
        self.rs = np.random.RandomState(self._seed)
        self.done = False
        self.steps = 0  # Count of steps taken in this episode
        # Set the button timer to zero (so button is immediately visible)
        self.buttons_timer = 0

        self.clear()
        # if no env params are provided and there is no list of params,
        # sample new random environment
        if env_params is None and self._param_list is None:
            self.build()

        # else if no env params are provided and there is a list of
        # params, choose the environment setting from the list of params.
        
        elif env_params is None:
            env_params = self._param_list[np.random.choice(len(self._param_list))]
            obj_dict = self.env_params_to_obj_dict(env_params)
            self.build(config_dict=obj_dict)

        # if env_params is provided, set environment accordingly
        else:
            obj_dict = self.env_params_to_obj_dict(env_params)
            self.build(config_dict=obj_dict)

        # Save the layout at reset
        self.reset_layout = deepcopy(self.layout)
        self.env_params = self.env_params_from_layout(self.layout)
        print(f"Env params set to {self.env_params}")

        cost = self.cost()
        assert cost['cost'] == 0, f'World has starting cost! {cost}'

        # Reset stateful parts of the environment
        self.first_reset = False  # Built our first world successfully

        # Return an observation
        obs_dict = {
            "state": self.obs(), 
            "env_params": self.env_params,
        }
        info = dict()
        if self.task == "cleanup":
            info["task_completion"] = self.cleanup_task_completion()
        if self.task == "cleanup_all":
            info["task_completion"] = [self.cleanup_task_completion(task_id) for task_id in common.DOMAIN_TASK_IDS["cleanup_all"]]
        return obs_dict, info
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.task == "cleanup":
            info["task_completion"] = self.cleanup_task_completion()
        if self.task == "cleanup_all":
            info["task_completion"] = [self.cleanup_task_completion(task_id) for task_id in common.DOMAIN_TASK_IDS["cleanup_all"]]
        obs_dict = {
            "state": obs, 
            "env_params": self.env_params,
        }
        return obs_dict, reward, done, info
