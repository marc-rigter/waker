#!/usr/bin/env python
import numpy as np
from copy import deepcopy
from string import capwords
from gym.envs.registration import register
import numpy as np
from envs.safety_gym.envs.suite import SafexpEnvBase

#========================================#
# Helper Class for Easy Gym Registration #
#========================================#
VERSION = 'v0'

class ReplayEnvBase(SafexpEnvBase):
    def __init__(self, name='', config={}):
        super().__init__(name=name, config=config, prefix="Replay")

    def register(self, name='', config={}, param_list=None, eval_cases=None):
        for robot_name, robot_config in self.robot_configs.items():
            # Default
            env_name = f'{self.prefix}-{robot_name.lower()}{self.name.lower() + name.lower()}-{VERSION}'
            reg_config = self.config.copy()
            reg_config.update(robot_config)
            reg_config.update(config)

            register(id=env_name,
                     entry_point='envs.safety_gym.envs.mujoco:ReplayEngine',
                     kwargs={'config': reg_config,
                             'param_list': param_list,
                             'eval_cases': eval_cases})

    def copy(self, name='', config={}):
        new_config = self.config.copy()
        new_config.update(config)
        return ReplayEnvBase(self.name + name, new_config)




bench_base = ReplayEnvBase('', {'sample_num_objects': True})


############################# Cleanup - All #########################################
num_arenas = 6
max_object_nums = [0, 1, 2, 3, 4, 5]
goal_width = 0.5
border = 0.0
arenas = list(range(num_arenas))
arenas_to_sample = arenas[:-1]
min_size = 0.7
max_size = 1.4
sizes = np.linspace(min_size, max_size, len(arenas)-1).tolist()
sizes.append(max_size)
placements_extents = [[-size+goal_width+border, -size+border, size-goal_width-border, size-border] for size in sizes]

cleanup = {
    'sample_blueobjs': True,
    'sample_greenobjs': True,
    'sample_arenas': True,
    'arenas': arenas,
    'arenas_to_sample': arenas_to_sample,
    'arena_sizes': sizes,
    'observe_goal_lidar': False,
    'arenas_max_object_nums': max_object_nums,
    'arenas_placements_extents': placements_extents,
    'arenas_robot_locations': [[] for i in range(num_arenas)],
    'left_goal_locs': [[-size, -2, -size+goal_width, 2] for size in sizes],
    'right_goal_locs': [[size-goal_width, -2, size, 2] for size in sizes],
    'reward_distance': 100.0,
    'blueobjs_keepout': 0.35,
    'greenobjs_keepout': 0.35,
}

all = {
    'task': 'cleanup_all',
}

eval_cases = {
    "OOD_5blocks": [np.array([5, 0, 5, 0, 0, 0]), np.array([5, 0, 5, 0, 0, 0]), \
                    np.array([5, 1, 4, 0, 0, 0]), np.array([5, 1, 4, 0, 0, 0]), \
                    np.array([5, 2, 3, 0, 0, 0]), np.array([5, 2, 3, 0, 0, 0]), \
                    np.array([5, 3, 2, 0, 0, 0]), np.array([5, 3, 2, 0, 0, 0]), \
                    np.array([5, 4, 1, 0, 0, 0]), np.array([5, 4, 1, 0, 0, 0]), \
                    np.array([5, 5, 0, 0, 0, 0]), np.array([5, 5, 0, 0, 0, 0]), \
                    np.array([5, 0, 5, 0, 0, 0]), np.array([5, 0, 5, 0, 0, 0]), \
                    np.array([5, 1, 4, 0, 0, 0]), np.array([5, 1, 4, 0, 0, 0]), \
                    np.array([5, 2, 3, 0, 0, 0]), np.array([5, 2, 3, 0, 0, 0]), \
                    np.array([5, 3, 2, 0, 0, 0]), np.array([5, 3, 2, 0, 0, 0]), \
                    np.array([5, 4, 1, 0, 0, 0]), np.array([5, 4, 1, 0, 0, 0]), \
                    np.array([5, 5, 0, 0, 0, 0]), np.array([5, 5, 0, 0, 0, 0])],

}

bench_cleanup_base = bench_base.copy('Cleanup', cleanup)
bench_cleanup_base.register('_All', all, eval_cases=eval_cases)
