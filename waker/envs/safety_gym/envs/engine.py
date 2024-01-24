#!/usr/bin/env python

import gym
import gym.spaces
import numpy as np
import itertools
import common
from PIL import Image
from copy import deepcopy
from collections import OrderedDict
import mujoco_py
from mujoco_py import MjViewer, MujocoException, const, MjRenderContextOffscreen

from envs.safety_gym.envs.world import World, Robot

import sys


# Distinct colors for different types of objects.
# For now this is mostly used for visualization.
# This also affects the vision observation, so if training from pixels.
COLOR_BOX = np.array([1, 1, 0, 1])
COLOR_BUTTON = np.array([1, .5, 0, 1])
COLOR_GOAL = np.array([0, 1, 0, 1])
COLOR_VASE = np.array([0, 1, 1, 1])
COLOR_HAZARD = np.array([0, 0, 1, 1])
COLOR_PILLAR = np.array([0, 0, 0, 1])
COLOR_WALL = np.array([.5, .5, .5, 1])
COLOR_GREMLIN = np.array([0.5, 0, 1, 1])
COLOR_CIRCLE = np.array([0, 1, 0, 1])
COLOR_RED = np.array([1, 0, 0, 1])
COLOR_GREEN = np.array([0, 1, 0, 1])
COLOR_BLUE = np.array([0, 0, 1, 1])
COLOR_YELLOW = np.array([1, 1, 0, 1])
COLOR_PURPLE = np.array([0.5, 0, 1, 1])
COLOR_CYAN = np.array([0, 1, 1, 1])

# Groups are a mujoco-specific mechanism for selecting which geom objects to "see"
# We use these for raycasting lidar, where there are different lidar types.
# These work by turning "on" the group to see and "off" all the other groups.
# See obs_lidar_natural() for more.
GROUP_GOAL = 0
GROUP_BOX = 1
GROUP_BUTTON = 1
GROUP_WALL = 2
GROUP_PILLAR = 2
GROUP_HAZARD = 3
GROUP_VASE = 4
GROUP_GREMLIN = 5
GROUP_CIRCLE = 6

# Constant for origin of world
ORIGIN_COORDINATES = np.zeros(3)

# Constant defaults for rendering frames for humans (not used for vision)
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256

class ResamplingError(AssertionError):
    ''' Raised when we fail to sample a valid distribution of objects or goals '''
    pass


def theta2vec(theta):
    ''' Convert an angle (in radians) to a unit vector in that angle around Z '''
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def quat2mat(quat):
    ''' Convert Quaternion to a 3x3 Rotation Matrix using mujoco '''
    q = np.array(quat, dtype='float64')
    m = np.zeros(9, dtype='float64')
    mujoco_py.functions.mju_quat2Mat(m, q)
    return m.reshape((3,3))


def quat2zalign(quat):
    ''' From quaternion, extract z_{ground} dot z_{body} '''
    # z_{body} from quaternion [a,b,c,d] in ground frame is:
    # [ 2bd + 2ac,
    #   2cd - 2ab,
    #   a**2 - b**2 - c**2 + d**2
    # ]
    # so inner product with z_{ground} = [0,0,1] is
    # z_{body} dot z_{ground} = a**2 - b**2 - c**2 + d**2
    a, b, c, d = quat
    return a**2 - b**2 - c**2 + d**2


class Engine(gym.Env, gym.utils.EzPickle):

    '''
    Engine: an environment-building tool for safe exploration research.

    The Engine() class entails everything to do with the tasks and safety 
    requirements of Safety Gym environments. An Engine() uses a World() object
    to interface to MuJoCo. World() configurations are inferred from Engine()
    configurations, so an environment in Safety Gym can be completely specified
    by the config dict of the Engine() object.

    '''

    # Default configuration (this should not be nested since it gets copied)
    DEFAULT = {
        'num_steps': 1000,  # Maximum number of environment steps in an episode
        'return_scale': None, # total reward for perfect episode
        'action_noise': 0.0,  # Magnitude of independent per-component gaussian action noise

        'placements_extents': [-2, -2, 2, 2],  # Placement limits (min X, min Y, max X, max Y)
        'placements_margin': 0.0,  # Additional margin added to keepout when placing objects

        # Floor
        'floor_display_mode': False,  # In display mode, the visible part of the floor is cropped

        # Robot
        'robot_placements': None,  # Robot placements list (defaults to full extents)
        'robot_locations': [],  # Explicitly place robot XY coordinate
        'robot_keepout': 0.2,  # Needs to be set to match the robot XML used
        'robot_base': 'xmls/car.xml',  # Which robot XML to use as the base
        'robot_rot': None,  # Override robot starting angle

        # Starting position distribution
        'randomize_layout': True,  # If false, set the random seed before layout to constant
        'build_resample': True,  # If true, rejection sample from valid environments
        'continue_goal': True,  # If true, draw a new goal after achievement
        'terminate_resample_failure': True,  # If true, end episode when resampling fails,
                                             # otherwise, raise a python exception.
        # TODO: randomize starting joint positions

        # Observation flags - some of these require other flags to be on
        # By default, only robot sensor observations are enabled.
        'observation_flatten': True,  # Flatten observation into a vector
        'observe_sensors': True,  # Observe all sensor data from simulator
        'observe_goal_dist': False,  # Observe the distance to the goal
        'observe_goal_comp': False,  # Observe a compass vector to the goal
        'observe_goal_lidar': False,  # Observe the goal with a lidar sensor
        'observe_box_comp': False,  # Observe the box with a compass
        'observe_box_lidar': False,  # Observe the box with a lidar
        'observe_circle': False,  # Observe the origin with a lidar
        'observe_remaining': False,  # Observe the fraction of steps remaining
        'observe_walls': False,  # Observe the walls with a lidar space
        'observe_hazards': False,  # Observe the vector from agent to hazards
        'observe_vases': False,  # Observe the vector from agent to vases
        'observe_pillars': False,  # Lidar observation of pillar object positions
        'observe_buttons': False,  # Lidar observation of button object positions
        'observe_gremlins': False,  # Gremlins are observed with lidar-like space
        'observe_vision': False,  # Observe vision from the robot
        # These next observations are unnormalized, and are only for debugging
        'observe_qpos': False,  # Observe the qpos of the world
        'observe_qvel': False,  # Observe the qvel of the robot
        'observe_ctrl': False,  # Observe the previous action
        'observe_freejoint': False,  # Observe base robot free joint
        'observe_com': False,  # Observe the center of mass of the robot

        # Render options
        'render_labels': False,
        'render_lidar_markers': True,
        'render_lidar_radius': 0.15, 
        'render_lidar_size': 0.025, 
        'render_lidar_offset_init': 0.5, 
        'render_lidar_offset_delta': 0.06, 

        # Vision observation parameters
        'vision_size': (60, 40),  # Size (width, height) of vision observation; gets flipped internally to (rows, cols) format
        'vision_render': True,  # Render vision observation in the viewer
        'vision_render_size': (300, 200),  # Size to render the vision in the viewer

        # Lidar observation parameters
        'lidar_num_bins': 10,  # Bins (around a full circle) for lidar sensing
        'lidar_max_dist': None,  # Maximum distance for lidar sensitivity (if None, exponential distance)
        'lidar_exp_gain': 1.0, # Scaling factor for distance in exponential distance lidar
        'lidar_type': 'pseudo',  # 'pseudo', 'natural', see self.obs_lidar()
        'lidar_alias': True,  # Lidar bins alias into each other

        # Compass observation parameters
        'compass_shape': 2,  # Set to 2 or 3 for XY or XYZ unit vector compass observation.

        # Task
        'task': 'cleanup_all',  # goal, button, push, x, z, circle, or none (for screenshots)

        # parameters for cleanup task
        'sample_blueobjs': False,
        'sample_greenobjs': False,
        'blueobjs_num': 0,
        'greenobjs_num': 0,
        'blueobjs_keepout': 0.25,
        'greenobjs_keepout': 0.25,
        'right_goal_locs': [[-2, 2, -1, 2]], # Goal location for blue objects (min X, min Y, max X, max Y)
        'left_goal_locs': [[1, -2, 2, 2]], # Goal location for green objects (min X, min Y, max X, max Y)
        'blueobjs_size': 0.2,
        'greenobjs_size': 0.2,
        'reward_task_completion': 0.5,

        # Params for domain randomisation 
        'sample_num_objects': False,  # whether to sample the number of objects between 0 and max.
        'sample_objects_prob': 1.0,  # probability of adding objects to environment
        'sample_arenas': False,
        'max_object_num': 0,
        'arenas_max_object_nums': [0],
        'arenas': [0],
        'arenas_to_sample': [0],
        'arena_sizes': [0.0],
        'arenas_placements_extents': [[-2, -2, 2, 2]],
        'arenas_goal_locations': [[]],
        'arenas_box_locations': [[]],
        'arenas_robot_locations': [[]],
        'sample_vases': False,
        'sample_hazards': False,
        'sample_gremlins': False,
        'sample_pillars': False,
        'sample_buttons': False,

        # Goal parameters
        'goal_placements': None,  # Placements where goal may appear (defaults to full extents)
        'goal_locations': [],  # Fixed locations to override placements
        'goal_keepout': 0.4,  # Keepout radius when placing goals
        'goal_size': 0.3,  # Radius of the goal area (if using task 'goal')

        # Box parameters (only used if task == 'push')
        'box_placements': None,  # Box placements list (defaults to full extents)
        'box_locations': [],  # Fixed locations to override placements
        'box_keepout': 0.2,  # Box keepout radius for placement
        'box_size': 0.2,  # Box half-radius size
        'box_density': 0.001,  # Box density
        'box_null_dist': 2, # Within box_null_dist * box_size radius of box, no box reward given

        # Reward is distance towards goal plus a constant for being within range of goal
        # reward_distance should be positive to encourage moving towards the goal
        # if reward_distance is 0, then the reward function is sparse
        'reward_distance': 1.0,  # Dense reward multiplied by the distance moved to the goal
        'reward_goal': 0.0,  # Sparse reward for being inside the goal area
        'reward_box_dist': 0.0,  # Dense reward for moving the robot towards the box
        'reward_box_goal': 1.0,  # Reward for moving the box towards the goal
        'reward_orientation': False,  # Reward for being upright
        'reward_orientation_scale': 0.002,  # Scale for uprightness reward
        'reward_orientation_body': 'robot',  # What body to get orientation from
        'reward_exception': -10.0,  # Reward when encoutering a mujoco exception
        'reward_x': 1.0,  # Reward for forward locomotion tests (vel in x direction)
        'reward_z': 1.0,  # Reward for standup tests (vel in z direction)
        'reward_circle': 1e-1,  # Reward for circle goal (complicated formula depending on pos and vel)
        'reward_clip': 10,  # Clip reward, last resort against physics errors causing magnitude spikes

        # Buttons are small immovable spheres, to the environment
        'buttons_num': 0,  # Number of buttons to add
        'buttons_placements': None,  # Buttons placements list (defaults to full extents)
        'buttons_locations': [],  # Fixed locations to override placements
        'buttons_keepout': 0.3,  # Buttons keepout radius for placement
        'buttons_size': 0.1,  # Size of buttons in the scene
        'buttons_cost': 1.0,  # Cost for pressing the wrong button, if constrain_buttons
        'buttons_resampling_delay': 10,  # Buttons have a timeout period (steps) before resampling

        # Circle parameters (only used if task == 'circle')
        'circle_radius': 1.5,

        # Sensor observations
        # Specify which sensors to add to observation space
        'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
        'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
        'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
        'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

        # Walls - barriers in the environment not associated with any constraint
        # NOTE: this is probably best to be auto-generated than manually specified
        'walls_num': 0,  # Number of walls
        'walls_placements': None,  # This should not be used
        'walls_locations': [],  # This should be used and length == walls_num
        'walls_keepout': 0.0,  # This should not be used
        'walls_size': 0.5,  # Should be fixed at fundamental size of the world

        # Constraints - flags which can be turned on
        # By default, no constraints are enabled, and all costs are indicator functions.
        'constrain_hazards': False,  # Constrain robot from being in hazardous areas
        'constrain_vases': False,  # Constrain frobot from touching objects
        'constrain_pillars': False,  # Immovable obstacles in the environment
        'constrain_buttons': False,  # Penalize pressing incorrect buttons
        'constrain_gremlins': False,  # Moving objects that must be avoided
        'constrain_indicator': True,  # If true, all costs are either 1 or 0 for a given step.

        # Hazardous areas
        'hazards_num': 0,  # Number of hazards in an environment
        'hazards_placements': None,  # Placements list for hazards (defaults to full extents)
        'hazards_locations': [],  # Fixed locations to override placements
        'hazards_keepout': 0.4,  # Radius of hazard keepout for placement
        'hazards_size': 0.3,  # Radius of hazards
        'hazards_cost': 1.0,  # Cost (per step) for violating the constraint

        # Vases (objects we should not touch)
        'vases_num': 0,  # Number of vases in the world
        'vases_placements': None,  # Vases placements list (defaults to full extents)
        'vases_locations': [],  # Fixed locations to override placements
        'vases_keepout': 0.15,  # Radius of vases keepout for placement
        'vases_size': 0.1,  # Half-size (radius) of vase object
        'vases_density': 0.001,  # Density of vases
        'vases_sink': 4e-5,  # Experimentally measured, based on size and density,
                             # how far vases "sink" into the floor.
        # Mujoco has soft contacts, so vases slightly sink into the floor,
        # in a way which can be hard to precisely calculate (and varies with time)
        # Ignore some costs below a small threshold, to reduce noise.
        'vases_contact_cost': 1.0,  # Cost (per step) for being in contact with a vase
        'vases_displace_cost': 0.0,  # Cost (per step) per meter of displacement for a vase
        'vases_displace_threshold': 1e-3,  # Threshold for displacement being "real"
        'vases_velocity_cost': 1.0,  # Cost (per step) per m/s of velocity for a vase
        'vases_velocity_threshold': 1e-4,  # Ignore very small velocities

        # Pillars (immovable obstacles we should not touch)
        'pillars_num': 0,  # Number of pillars in the world
        'pillars_placements': None,  # Pillars placements list (defaults to full extents)
        'pillars_locations': [],  # Fixed locations to override placements
        'pillars_keepout': 0.3,  # Radius for placement of pillars
        'pillars_size': 0.2,  # Half-size (radius) of pillar objects
        'pillars_height': 0.1,  # Half-height of pillars geoms
        'pillars_cost': 1.0,  # Cost (per step) for being in contact with a pillar

        # Gremlins (moving objects we should avoid)
        'gremlins_num': 0,  # Number of gremlins in the world
        'gremlins_placements': None,  # Gremlins placements list (defaults to full extents)
        'gremlins_locations': [],  # Fixed locations to override placements
        'gremlins_keepout': 0.5,  # Radius for keeping out (contains gremlin path)
        'gremlins_travel': 0.3,  # Radius of the circle traveled in
        'gremlins_size': 0.1,  # Half-size (radius) of gremlin objects
        'gremlins_density': 0.001,  # Density of gremlins
        'gremlins_contact_cost': 1.0,  # Cost for touching a gremlin
        'gremlins_dist_threshold': 0.2,  # Threshold for cost for being too close
        'gremlins_dist_cost': 1.0,  # Cost for being within distance threshold

        # Frameskip is the number of physics simulation steps per environment step
        # Frameskip is sampled as a binomial distribution
        # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
        'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip)
        'frameskip_binom_p': 1.0,  # Probability of trial return (controls distribution)

        '_seed': None,  # Random state seed (avoid name conflict with self.seed)
    }

    def __init__(self, config={}):
        # First, parse configuration. Important note: LOTS of stuff happens in
        # parse, and many attributes of the class get set through setattr. If you
        # are trying to track down where an attribute gets initially set, and 
        # can't find it anywhere else, it's probably set via the config dict
        # and this parse function.
        self.parse(config)
        gym.utils.EzPickle.__init__(self, config=config)

        # Load up a simulation of the robot, just to figure out observation space
        self.robot = Robot(self.robot_base)

        self.action_space = gym.spaces.Box(-1, 1, (self.robot.nu,), dtype=np.float32)
        self.build_observation_space()

        self.viewer = None
        self.world = None
        self.clear()

        self.seed(self._seed)
        self.done = True

        self.configurable_types = ["greenobj", "blueobj", "hazard", "pillar", "vase"]

    def parse(self, config):
        ''' Parse a config dict - see self.DEFAULT for description '''
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    @property
    def sim(self):
        ''' Helper to get the world's simulation instance '''
        return self.world.sim

    @property
    def model(self):
        ''' Helper to get the world's model instance '''
        return self.sim.model

    @property
    def data(self):
        ''' Helper to get the world's simulation data instance '''
        return self.sim.data

    @property
    def robot_pos(self):
        ''' Helper to get current robot position '''
        return self.data.get_body_xpos('robot').copy()

    @property
    def goal_pos(self):
        ''' Helper to get goal position from layout '''
        if self.task in ['goal', 'push']:
            return self.data.get_body_xpos('goal').copy()
        elif self.task == 'button':
            return self.data.get_body_xpos(f'button{self.goal_button}').copy()
        elif self.task == 'circle':
            return ORIGIN_COORDINATES
        elif self.task == 'none':
            return np.zeros(2)  # Only used for screenshots
        elif self.task == 'cleanup' or self.task == 'cleanup_all':
            pass
        else:
            raise ValueError(f'Invalid task {self.task}')

    @property
    def box_pos(self):
        ''' Helper to get the box position '''
        return self.data.get_body_xpos('box').copy()

    @property
    def buttons_pos(self):
        ''' Helper to get the list of button positions '''
        return [self.data.get_body_xpos(f'button{i}').copy() for i in range(self.num_objects_in_layout("button", self.layout))]

    @property
    def vases_pos(self):
        ''' Helper to get the list of vase positions '''
        return [self.data.get_body_xpos(f'vase{p}').copy() for p in range(self.num_objects_in_layout("vase", self.layout))]

    @property
    def gremlins_obj_pos(self):
        ''' Helper to get the current gremlin position '''
        return [self.data.get_body_xpos(f'gremlin{i}obj').copy() for i in range(self.num_objects_in_layout("gremlin", self.layout))]

    @property
    def pillars_pos(self):
        ''' Helper to get list of pillar positions '''
        return [self.data.get_body_xpos(f'pillar{i}').copy() for i in range(self.num_objects_in_layout("pillar", self.layout))]

    @property
    def hazards_pos(self):
        ''' Helper to get the hazards positions from layout '''
        return [self.data.get_body_xpos(f'hazard{i}').copy() for i in range(self.num_objects_in_layout("hazard", self.layout))]

    @property
    def walls_pos(self):
        ''' Helper to get the hazards positions from layout '''
        return [self.data.get_body_xpos(f'wall{i}').copy() for i in range(self.num_objects_in_layout("wall", self.layout))]
    
    @property
    def greenobjs_pos(self):
        ''' Helper to get the greenobjs positions from layout '''
        return [self.data.get_body_xpos(f'greenobj{i}').copy() for i in range(self.num_objects_in_layout("greenobj", self.layout))]
    
    @property
    def leftgoal_pos(self):
        ''' Helper to get the leftgoal positions from layout '''
        return self.data.get_body_xpos('leftgoal').copy()
    
    @property
    def rightgoal_pos(self):
        ''' Helper to get the rightgoal positions from layout '''
        return self.data.get_body_xpos('rightgoal').copy()
    
    @property
    def blueobjs_pos(self):
        ''' Helper to get the blueobjs positions from layout '''
        return [self.data.get_body_xpos(f'blueobj{i}').copy() for i in range(self.num_objects_in_layout("blueobj", self.layout))]

    def build_observation_space(self):
        ''' Construct observtion space.  Happens only once at during __init__ '''
        obs_space_dict = OrderedDict()  # See self.obs()

        if self.observe_freejoint:
            obs_space_dict['freejoint'] = gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)
        if self.observe_com:
            obs_space_dict['com'] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        if self.observe_sensors:
            for sensor in self.sensors_obs:  # Explicitly listed sensors
                dim = self.robot.sensor_dim[sensor]
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
            # Velocities don't have wraparound effects that rotational positions do
            # Wraparounds are not kind to neural networks
            # Whereas the angle 2*pi is very close to 0, this isn't true in the network
            # In theory the network could learn this, but in practice we simplify it
            # when the sensors_angle_components switch is enabled.
            for sensor in self.robot.hinge_vel_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            for sensor in self.robot.ballangvel_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
            # Angular positions have wraparound effects, so output something more friendly
            if self.sensors_angle_components:
                # Single joints are turned into sin(x), cos(x) pairs
                # These should be easier to learn for neural networks,
                # Since for angles, small perturbations in angle give small differences in sin/cos
                for sensor in self.robot.hinge_pos_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
                # Quaternions are turned into 3x3 rotation matrices
                # Quaternions have a wraparound issue in how they are normalized,
                # where the convention is to change the sign so the first element to be positive.
                # If the first element is close to 0, this can mean small differences in rotation
                # lead to large differences in value as the latter elements change sign.
                # This also means that the first element of the quaternion is not expectation zero.
                # The SO(3) rotation representation would be a good replacement here,
                # since it smoothly varies between values in all directions (the property we want),
                # but right now we have very little code to support SO(3) roatations.
                # Instead we use a 3x3 rotation matrix, which if normalized, smoothly varies as well.
                for sensor in self.robot.ballquat_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3, 3), dtype=np.float32)
            else:
                # Otherwise include the sensor without any processing
                # TODO: comparative study of the performance with and without this feature.
                for sensor in self.robot.hinge_pos_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
                for sensor in self.robot.ballquat_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        if self.task == 'push':
            if self.observe_box_comp:
                obs_space_dict['box_compass'] = gym.spaces.Box(-1.0, 1.0, (self.compass_shape,), dtype=np.float32)
            if self.observe_box_lidar:
                obs_space_dict['box_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_goal_dist:
            obs_space_dict['goal_dist'] = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        if self.observe_goal_comp:
            obs_space_dict['goal_compass'] = gym.spaces.Box(-1.0, 1.0, (self.compass_shape,), dtype=np.float32)
        if self.observe_goal_lidar:
            obs_space_dict['goal_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.task == 'circle' and self.observe_circle:
            obs_space_dict['circle_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_remaining:
            obs_space_dict['remaining'] = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        if self.walls_num and self.observe_walls:
            obs_space_dict['walls_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_hazards:
            obs_space_dict['hazards_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_vases:
            obs_space_dict['vases_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.gremlins_num and self.observe_gremlins:
            obs_space_dict['gremlins_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.pillars_num and self.observe_pillars:
            obs_space_dict['pillars_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.buttons_num and self.observe_buttons:
            obs_space_dict['buttons_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_qpos:
            obs_space_dict['qpos'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nq,), dtype=np.float32)
        if self.observe_qvel:
            obs_space_dict['qvel'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nv,), dtype=np.float32)
        if self.observe_ctrl:
            obs_space_dict['ctrl'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nu,), dtype=np.float32)
        if self.observe_vision:
            width, height = self.vision_size
            rows, cols = height, width
            self.vision_size = (rows, cols)
            obs_space_dict['vision'] = gym.spaces.Box(0, 1.0, self.vision_size + (3,), dtype=np.float32)
        # Flatten it ourselves
        self.obs_space_dict = obs_space_dict
        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict(obs_space_dict)

    def toggle_observation_space(self):
        self.observation_flatten = not(self.observation_flatten)
        self.build_observation_space()

    def placements_from_location(self, location, keepout):
        ''' Helper to get a placements list from a given location and keepout '''
        x, y = location
        return [(x - keepout, y - keepout, x + keepout, y + keepout)]

    def placements_dict_from_object(self, object_name, obj_dict=None):
        ''' Get the placements dict subset just for a given object name.

        Args:
            object_name
            sample_num: if True the number of objects to be added is sampled uniformly
            between 0 and the number specified.
        '''
        placements_dict = {}
        if hasattr(self, object_name + 's_num'):  # Objects with multiplicity
            plural_name = object_name + 's'
            object_fmt = object_name + '{i}'
            object_keepout = getattr(self, plural_name + '_keepout')

            if object_name == 'wall' and self.sample_arenas:
                object_num = 8
                wall_size = self.arena_sizes[self.arena]
                dist = 2 * wall_size
                object_locations = [np.array((-dist, -dist)), 
                                    np.array((dist, -dist)), 
                                    np.array((dist, dist)), 
                                    np.array((-dist, dist)), 
                                    np.array((-dist, 0)), 
                                    np.array((0, -dist)),
                                    np.array((0, dist)),
                                    np.array((dist, 0)),
                ]
                object_placements = None
            else:
                object_num = getattr(self, plural_name + '_num', None)
                object_locations = getattr(self, plural_name + '_locations', [])
                object_placements = getattr(self, plural_name + '_placements', None)
            
                # use number in object dict if provided
                if obj_dict is not None:
                    if object_name in obj_dict.keys():
                        object_num = obj_dict[object_name]

                

        else:  # Unique objects
            object_fmt = object_name
            object_num = 1
            if self.sample_arenas:
                object_locations = getattr(self, "arenas_" + object_name + '_locations', [[] for _ in range(len(self.arenas))])
                object_placements = getattr(self, "arenas_" + object_name + '_placements', [None for _ in range(len(self.arenas))])
                object_locations = object_locations[self.arena]
                object_placements = object_placements[self.arena]
                object_keepout = getattr(self, object_name + '_keepout')
            else:
                object_locations = getattr(self, object_name + '_locations', [])
                object_placements = getattr(self, object_name + '_placements', None)
                object_keepout = getattr(self, object_name + '_keepout')

        for i in range(object_num):
            if i < len(object_locations):
                x, y = object_locations[i]
                k = object_keepout + 1e-9  # Epsilon to account for numerical issues
                placements = [(x - k, y - k, x + k, y + k)]
            else:
                placements = object_placements
            placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
        return placements_dict

    def build_placements_dict(self, config_dict=None):
        ''' Build a dict of placements.  Happens once during __init__. '''
        # Dictionary is map from object name -> tuple of (placements list, keepout)
        
        
        if config_dict is None:
            config_dict = self.sample_config_dict()

        if self.sample_arenas:
            self.arena = config_dict["arena"]
            self.placement_extent = self.arenas_placements_extents[self.arena]
        placements = {}
        placements.update(self.placements_dict_from_object('robot'))
        placements.update(self.placements_dict_from_object('wall'))

        if self.task in ['goal', 'push']:
            placements.update(self.placements_dict_from_object('goal'))
        if self.task == 'push':
            placements.update(self.placements_dict_from_object('box'))
        if self.task == 'button' or self.buttons_num or self.sample_buttons:
            placements.update(self.placements_dict_from_object('button', config_dict))
        if self.hazards_num or self.sample_hazards:
            placements.update(self.placements_dict_from_object('hazard', config_dict))
        if self.vases_num or self.sample_vases:
            placements.update(self.placements_dict_from_object('vase', config_dict))
        if self.pillars_num or self.sample_pillars: 
            placements.update(self.placements_dict_from_object('pillar', config_dict))
        if self.gremlins_num or self.sample_gremlins: 
            placements.update(self.placements_dict_from_object('gremlin', config_dict))
        if self.blueobjs_num or self.sample_blueobjs: 
            placements.update(self.placements_dict_from_object('blueobj', config_dict))
        if self.greenobjs_num or self.sample_greenobjs: 
            placements.update(self.placements_dict_from_object('greenobj', config_dict))

        self.placements = placements

    def get_objects_to_sample(self):
        to_sample = []
        if self.sample_vases:
            to_sample.append("vase")
        if self.sample_hazards:
            to_sample.append("hazard")
        if self.sample_pillars:
            to_sample.append("pillar")
        if self.sample_gremlins:
            to_sample.append("gremlin")
        if self.sample_buttons:
            to_sample.append("button")
        if self.sample_blueobjs:
            to_sample.append("blueobj")
        if self.sample_greenobjs:
            to_sample.append("greenobj")
        return to_sample

    def sample_config_dict(self):
        if self.sample_arenas:
            arena = np.random.choice(self.arenas_to_sample)
            max_object_num = self.arenas_max_object_nums[arena]
        else:
            arena = 0
            max_object_num = self.max_object_num

        config_dict = {obj: 0 for obj in self.configurable_types}
        config_dict["arena"] = arena

        if self.sample_num_objects and np.random.uniform() < self.sample_objects_prob:
            object_num = self.rs.choice(max_object_num + 1)
            if object_num > 0:
                obj_types_to_sample = self.get_objects_to_sample()
                all_combos = list(itertools.combinations_with_replacement(obj_types_to_sample, object_num))
                sampled_obj = all_combos[np.random.choice(len(all_combos))]
                [config_dict.update({obj: sampled_obj.count(obj)}) for obj in self.configurable_types]
        return config_dict

    def seed(self, seed=None):
        ''' Set internal random state seeds '''
        self._seed = np.random.randint(2**32) if seed is None else seed

    def build_layout(self):
        ''' Rejection sample a placement of objects to find a layout. '''
        if not self.randomize_layout:
            self.rs = np.random.RandomState(0)

        for _ in range(50000):
            if self.sample_layout():
                break
        else:
            raise ResamplingError('Failed to sample layout of objects')

    def sample_layout(self):
        ''' Sample a single layout, returning True if successful, else False. '''

        def placement_is_valid(xy, layout):
            for other_name, other_xy in layout.items():
                other_keepout = self.placements[other_name][1]
                dist = np.sqrt(np.sum(np.square(xy - other_xy)))
                if dist < other_keepout + self.placements_margin + keepout:
                    return False
            return True

        layout = {}
        for name, (placements, keepout) in self.placements.items():
            conflicted = True
            for _ in range(100):
                xy = self.draw_placement(placements, keepout, self.placement_extent)
                if placement_is_valid(xy, layout):
                    conflicted = False
                    break
            if conflicted:
                return False
            layout[name] = xy
        self.layout = layout
        return True

    def constrain_placement(self, placement, keepout):
        ''' Helper function to constrain a single placement by the keepout radius '''
        xmin, ymin, xmax, ymax = placement
        return (xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout)

    def draw_placement(self, placements, keepout, extent):
        ''' 
        Sample an (x,y) location, based on potential placement areas.

        Summary of behavior: 

        'placements' is a list of (xmin, xmax, ymin, ymax) tuples that specify 
        rectangles in the XY-plane where an object could be placed. 

        'keepout' describes how much space an object is required to have
        around it, where that keepout space overlaps with the placement rectangle.

        To sample an (x,y) pair, first randomly select which placement rectangle
        to sample from, where the probability of a rectangle is weighted by its
        area. If the rectangles are disjoint, there's an equal chance the (x,y) 
        location will wind up anywhere in the placement space. If they overlap, then
        overlap areas are double-counted and will have higher density. This allows
        the user some flexibility in building placement distributions. Finally, 
        randomly draw a uniform point within the selected rectangle.

        '''
        if placements is None:
            choice = self.constrain_placement(extent, keepout)
        else:
            # Draw from placements according to placeable area
            constrained = []
            for placement in placements:
                xmin, ymin, xmax, ymax = self.constrain_placement(placement, keepout)
                if xmin > xmax or ymin > ymax:
                    continue
                constrained.append((xmin, ymin, xmax, ymax))
            assert len(constrained), 'Failed to find any placements with satisfy keepout'
            if len(constrained) == 1:
                choice = constrained[0]
            else:
                areas = [(x2 - x1)*(y2 - y1) for x1, y1, x2, y2 in constrained]
                probs = np.array(areas) / np.sum(areas)
                choice = constrained[self.rs.choice(len(constrained), p=probs)]
        xmin, ymin, xmax, ymax = choice
        return np.array([self.rs.uniform(xmin, xmax), self.rs.uniform(ymin, ymax)])

    def random_rot(self):
        ''' Use internal random state to get a random rotation in radians '''
        return self.rs.uniform(0, 2 * np.pi)
    
    def num_objects_in_layout(self, object_type, layout):
        ''' Count the number of occurences of this type of object in the layout.'''
        num = 0 
        for obj_name in layout.keys():
            if object_type in obj_name:
                num += 1
        return num

    def build_world_config(self):
        ''' Create a world_config from our own config '''
        # TODO: parse into only the pieces we want/need
        world_config = {}

        world_config['robot_base'] = self.robot_base
        world_config['robot_xy'] = self.layout['robot']
        if self.robot_rot is None:
            world_config['robot_rot'] = self.random_rot()
        else:
            world_config['robot_rot'] = float(self.robot_rot)

        if self.floor_display_mode:
            floor_size = max(self.placements_extents)
            world_config['floor_size'] = [floor_size + .1, floor_size + .1, 1]

        #if not self.observe_vision:
        #    world_config['render_context'] = -1  # Hijack this so we don't create context
        world_config['observe_vision'] = self.observe_vision

        # Extra objects to add to the scene
        world_config['objects'] = {}
        vases_num = self.num_objects_in_layout("vase", self.layout)
        if vases_num:
            for i in range(vases_num):
                name = f'vase{i}'
                object = {'name': name,
                          'size': np.ones(3) * self.vases_size,
                          'type': 'box',
                          'density': self.vases_density,
                          'pos': np.r_[self.layout[name], self.vases_size - self.vases_sink],
                          'rot': self.random_rot(),
                          'group': GROUP_VASE,
                          'rgba': COLOR_VASE}
                world_config['objects'][name] = object

        greenobjs_num = self.num_objects_in_layout("greenobj", self.layout)
        if greenobjs_num:
            for i in range(greenobjs_num):
                name = f'greenobj{i}'
                object = {'name': name,
                          'size': np.ones(3) * self.greenobjs_size,
                          'type': 'box',
                          'density': self.vases_density,
                          'pos': np.r_[self.layout[name], self.greenobjs_size - self.vases_sink],
                          'rot': self.random_rot(),
                          'group': GROUP_VASE,
                          'rgba': COLOR_GREEN}
                world_config['objects'][name] = object

        blueobjs_num = self.num_objects_in_layout("blueobj", self.layout)
        if blueobjs_num:
            for i in range(blueobjs_num):
                name = f'blueobj{i}'
                object = {'name': name,
                          'size': np.ones(3) * self.blueobjs_size,
                          'type': 'box',
                          'density': self.vases_density,
                          'pos': np.r_[self.layout[name], self.blueobjs_size - self.vases_sink],
                          'rot': self.random_rot(),
                          'group': GROUP_VASE,
                          'rgba': COLOR_BLUE}
                world_config['objects'][name] = object

        gremlins_num = self.num_objects_in_layout("gremlin", self.layout)
        if gremlins_num:
            self._gremlins_rots = dict()
            for i in range(gremlins_num):
                name = f'gremlin{i}obj'
                self._gremlins_rots[i] = self.random_rot()
                object = {'name': name,
                          'size': np.ones(3) * self.gremlins_size,
                          'type': 'box',
                          'density': self.gremlins_density,
                          'pos': np.r_[self.layout[name.replace('obj', '')], self.gremlins_size],
                          'rot': self._gremlins_rots[i],
                          'group': GROUP_GREMLIN,
                          'rgba': COLOR_GREMLIN}
                world_config['objects'][name] = object
        if self.task == 'push':
            object = {'name': 'box',
                      'type': 'box',
                      'size': np.ones(3) * self.box_size,
                      'pos': np.r_[self.layout['box'], self.box_size],
                      'rot': self.random_rot(),
                      'density': self.box_density,
                      'group': GROUP_BOX,
                      'rgba': COLOR_BOX}
            world_config['objects']['box'] = object

        # Extra geoms (immovable objects) to add to the scene
        world_config['geoms'] = {}
        if self.task in ['goal', 'push']:
            geom = {'name': 'goal',
                    'size': [self.goal_size, self.goal_size / 2],
                    'pos': np.r_[self.layout['goal'], self.goal_size / 2 + 1e-2],
                    'rot': self.random_rot(),
                    'type': 'cylinder',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': GROUP_GOAL,
                    'rgba': COLOR_GOAL * [1, 1, 1, 0.25]}  # transparent
            world_config['geoms']['goal'] = geom

        if self.task in ['cleanup', 'cleanup_all']:
            loc = self.left_goal_locs[self.arena]
            geom = {'name': 'leftgoal',
                    'size': [(loc[2] - loc[0])/2, (loc[3] - loc[1])/2, 0.04],
                    'pos': [(loc[2] + loc[0])/2, (loc[3] + loc[1])/2, 0.02],
                    'rot': 0,
                    'type': 'box',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': GROUP_GOAL,
                    'rgba': COLOR_GREEN * [1, 1, 1, 0.1]}  # transparent
            world_config['geoms']['leftgoal'] = geom

            loc = self.right_goal_locs[self.arena]
            geom = {'name': 'rightgoal',
                    'size': [(loc[2] - loc[0])/2, (loc[3] - loc[1])/2, 0.04],
                    'pos': [(loc[2] + loc[0])/2, (loc[3] + loc[1])/2, 0.02],
                    'rot': 0,
                    'type': 'box',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': GROUP_GOAL,
                    'rgba': COLOR_BLUE * [1, 1, 1, 0.1]}  # transparent
            world_config['geoms']['rightgoal'] = geom

        hazards_num = self.num_objects_in_layout("hazard", self.layout)
        if hazards_num:
            for i in range(hazards_num):
                name = f'hazard{i}'
                geom = {'name': name,
                        'size': [self.hazards_size, 1e-2],#self.hazards_size / 2],
                        'pos': np.r_[self.layout[name], 2e-2],#self.hazards_size / 2 + 1e-2],
                        'rot': self.random_rot(),
                        'type': 'cylinder',
                        'contype': 0,
                        'conaffinity': 0,
                        'group': GROUP_HAZARD,
                        'rgba': COLOR_HAZARD * [1, 1, 1, 1.0]} #0.1]}  # transparent
                world_config['geoms'][name] = geom

        pillars_num = self.num_objects_in_layout("pillar", self.layout)
        if pillars_num:
            for i in range(pillars_num):
                name = f'pillar{i}'
                geom = {'name': name,
                        'size': [self.pillars_size, self.pillars_height],
                        'pos': np.r_[self.layout[name], self.pillars_height],
                        'rot': self.random_rot(),
                        'type': 'cylinder',
                        'group': GROUP_PILLAR,
                        'rgba': COLOR_PILLAR}
                world_config['geoms'][name] = geom

        walls_num = self.num_objects_in_layout("wall", self.layout)
        if self.sample_arenas:
            wall_size = self.arena_sizes[self.arena]
        else:
            wall_size = self.walls_size
        size = np.array([1, 1, 1]) * wall_size
        size[2] = 0.15
        if walls_num:
            for i in range(walls_num):
                name = f'wall{i}'
                geom = {'name': name,
                        'size': size,
                        'pos': np.r_[self.layout[name], 0.1*wall_size],
                        'rot': 0,
                        'type': 'box',
                        'group': GROUP_WALL,
                        'rgba': COLOR_WALL}
                world_config['geoms'][name] = geom

        buttons_num = self.num_objects_in_layout("button", self.layout)
        if buttons_num:
            for i in range(buttons_num):
                name = f'button{i}'
                geom = {'name': name,
                        'size': np.ones(3) * self.buttons_size,
                        'pos': np.r_[self.layout[name], self.buttons_size],
                        'rot': self.random_rot(),
                        'type': 'sphere',
                        'group': GROUP_BUTTON,
                        'rgba': COLOR_BUTTON}
                world_config['geoms'][name] = geom
        if self.task == 'circle':
            geom = {'name': 'circle',
                    'size': np.array([self.circle_radius, 1e-2]),
                    'pos': np.array([0, 0, 2e-2]),
                    'rot': 0,
                    'type': 'cylinder',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': GROUP_CIRCLE,
                    'rgba': COLOR_CIRCLE * [1, 1, 1, 0.1]}
            world_config['geoms']['circle'] = geom


        # Extra mocap bodies used for control (equality to object of same name)
        world_config['mocaps'] = {}
        gremlins_num = self.num_objects_in_layout("gremlin", self.layout)
        if gremlins_num:
            for i in range(gremlins_num):
                name = f'gremlin{i}mocap'
                mocap = {'name': name,
                         'size': np.ones(3) * self.gremlins_size,
                         'type': 'box',
                         'pos': np.r_[self.layout[name.replace('mocap', '')], self.gremlins_size],
                         'rot': self._gremlins_rots[i],
                         'group': GROUP_GREMLIN,
                         'rgba': np.array([1, 1, 1, .1]) * COLOR_GREMLIN}
                         #'rgba': np.array([1, 1, 1, 0]) * COLOR_GREMLIN}
                world_config['mocaps'][name] = mocap

        return world_config

    def clear(self):
        ''' Reset internal state for building '''
        self.layout = None

    def build_goal(self):
        ''' Build a new goal position, maybe with resampling due to hazards '''
        if self.task == 'goal':
            self.build_goal_position()
            self.last_dist_goal = self.dist_goal()
        elif self.task == 'push':
            self.build_goal_position()
            self.last_dist_goal = self.dist_goal()
            self.last_dist_box = self.dist_box()
            self.last_box_goal = self.dist_box_goal()
        elif self.task == 'button':
            assert self.buttons_num > 0, 'Must have at least one button'
            self.build_goal_button()
            self.last_dist_goal = self.dist_goal()
        elif self.task in ['x', 'z']:
            self.last_robot_com = self.world.robot_com()
        elif self.task in ["cleanup"]:
            self.last_dist_obj_goals = self.dist_obj_goals()
        elif self.task in ["cleanup_all"]:
            self.last_dist_obj_goals = {
                task: self.dist_obj_goals(task) for task in common.DOMAIN_TASK_IDS["cleanup_all"]
            }
        elif self.task in ['circle', 'none']:
            pass
        else:
            raise ValueError(f'Invalid task {self.task}')

    def sample_goal_position(self):
        ''' Sample a new goal position and return True, else False if sample rejected '''
        placements, keepout = self.placements['goal']
        goal_xy = self.draw_placement(placements, keepout)
        for other_name, other_xy in self.layout.items():
            other_keepout = self.placements[other_name][1]
            dist = np.sqrt(np.sum(np.square(goal_xy - other_xy)))
            if dist < other_keepout + self.placements_margin + keepout:
                return False
        self.layout['goal'] = goal_xy
        return True

    def build_goal_position(self):
        ''' Build a new goal position, maybe with resampling due to hazards '''
        # Resample until goal is compatible with layout
        if 'goal' in self.layout:
            return
        for _ in range(10000):  # Retries
            if self.sample_goal_position():
                break
        else:
            raise ResamplingError('Failed to generate goal')
        # Move goal geom to new layout position
        self.world_config_dict['geoms']['goal']['pos'][:2] = self.layout['goal']
        #self.world.rebuild(deepcopy(self.world_config_dict))
        #self.update_viewer_sim = True
        goal_body_id = self.sim.model.body_name2id('goal')
        self.sim.model.body_pos[goal_body_id][:2] = self.layout['goal']
        self.sim.forward()

    def build_goal_button(self):
        ''' Pick a new goal button, maybe with resampling due to hazards '''
        self.goal_button = self.rs.choice(self.buttons_num)

    def build(self, config_dict=None):
        ''' Build a new physics simulation environment '''

        # sample number of objects
        self.build_placements_dict(config_dict)

        # Sample object positions
        self.build_layout()

        # Build the underlying physics world
        self.world_config_dict = self.build_world_config()

        if self.world is None:
            self.world = World(self.world_config_dict)
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_config_dict, state=False)
        # Redo a small amount of work, and setup initial goal state
        self.build_goal()

        # Save last action
        self.last_action = np.zeros(self.action_space.shape)

        # Save last subtree center of mass
        self.last_subtreecom = self.world.get_sensor('subtreecom')

    def reset(self):
        ''' Reset the physics simulation and return observation '''
        self._seed += 1  # Increment seed
        self.rs = np.random.RandomState(self._seed)
        self.done = False
        self.steps = 0  # Count of steps taken in this episode
        # Set the button timer to zero (so button is immediately visible)
        self.buttons_timer = 0

        self.clear()
        self.build()
        # Save the layout at reset
        self.reset_layout = deepcopy(self.layout)

        cost = self.cost()
        assert cost['cost'] == 0, f'World has starting cost! {cost}'

        # Reset stateful parts of the environment
        self.first_reset = False  # Built our first world successfully

        # Return an observation
        return self.obs()

    def dist_goal(self):
        ''' Return the distance from the robot to the goal XY position '''
        return self.dist_xy(self.goal_pos)

    def dist_box(self):
        ''' Return the distance from the robot to the box (in XY plane only) '''
        assert self.task == 'push', f'invalid task {self.task}'
        return np.sqrt(np.sum(np.square(self.box_pos - self.world.robot_pos())))
    
    def get_desired_goals(self, task):
        ''' Returns the desired goal locations for each object based on the task '''
        if task == "sort":
            return {"green": "left", "blue": "right"}
        elif task == "sort-reversed":
            return {"green": "right", "blue": "left"}
        elif task == "push":
            return {"green": "right", "blue": "right"}
        else:
            raise NotImplementedError

    def dist_obj_goals(self, task="sort"):
        ''' Returns the distance of each object to its goal location in the x-direction'''
        dists = []
        desired_goals = self.get_desired_goals(task)

        if desired_goals["green"] == "left":
            green_goal_pos = self.leftgoal_pos
        else:
            green_goal_pos = self.rightgoal_pos
        if desired_goals["blue"] == "left":
            blue_goal_pos = self.leftgoal_pos
        else:
            blue_goal_pos = self.rightgoal_pos

        for green_obj_pos in self.greenobjs_pos:
            xdist = np.abs(green_obj_pos[0] - green_goal_pos[0])
            dists.append(xdist)

        for blue_obj_pos in self.blueobjs_pos:
            xdist = np.abs(blue_obj_pos[0] - blue_goal_pos[0])
            dists.append(xdist)
        return dists

    def cleanup_task_completion(self, task="sort"):
        ''' Returns whether the proportion of the cleanup task completed '''
        total = self.num_objects_in_layout("blueobj", self.layout)
        total += self.num_objects_in_layout("greenobj", self.layout)

        # if there are no objects to clean up
        if total == 0:
            return 1
        
        # set the goal location according to cleanup task
        desired_goals = self.get_desired_goals(task)
        if desired_goals["green"] == "left":
            green_goal_loc = self.left_goal_locs[self.arena]
        else:
            green_goal_loc = self.right_goal_locs[self.arena]
        if desired_goals["blue"] == "left":
            blue_goal_loc = self.left_goal_locs[self.arena]
        else:
            blue_goal_loc = self.right_goal_locs[self.arena]

        # complete if within goal
        complete = 0
        for green_obj_pos in self.greenobjs_pos:
            if (green_obj_pos[0] > green_goal_loc[0]) and (green_obj_pos[0] < green_goal_loc[2]) \
                and (green_obj_pos[1] > green_goal_loc[1]) and (green_obj_pos[1] < green_goal_loc[3]):
                complete += 1

        for blue_obj_pos in self.blueobjs_pos:
            if (blue_obj_pos[0] > blue_goal_loc[0]) and (blue_obj_pos[0] < blue_goal_loc[2]) \
                and (blue_obj_pos[1] > blue_goal_loc[1]) and (blue_obj_pos[1] < blue_goal_loc[3]):
                complete += 1

        return complete/total

    def dist_box_goal(self):
        ''' Return the distance from the box to the goal XY position '''
        assert self.task == 'push', f'invalid task {self.task}'
        return np.sqrt(np.sum(np.square(self.box_pos - self.goal_pos)))

    def dist_xy(self, pos):
        ''' Return the distance from the robot to an XY position '''
        pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]
        robot_pos = self.world.robot_pos()
        return np.sqrt(np.sum(np.square(pos - robot_pos[:2])))

    def world_xy(self, pos):
        ''' Return the world XY vector to a position from the robot '''
        assert pos.shape == (2,)
        return pos - self.world.robot_pos()[:2]

    def ego_xy(self, pos):
        ''' Return the egocentric XY vector to a position from the robot '''
        assert pos.shape == (2,), f'Bad pos {pos}'
        robot_3vec = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        world_3vec = pos_3vec - robot_3vec
        return np.matmul(world_3vec, robot_mat)[:2]  # only take XY coordinates

    def obs_compass(self, pos):
        '''
        Return a robot-centric compass observation of a list of positions.

        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.

        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        '''
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        # Get ego vector in world frame
        vec = pos - self.world.robot_pos()
        # Rotate into frame
        vec = np.matmul(vec, self.world.robot_mat())
        # Truncate
        vec = vec[:self.compass_shape]
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        assert vec.shape == (self.compass_shape,), f'Bad vec {vec}'
        return vec

    def obs_vision(self):
        ''' Return pixels from the robot camera '''
        # Get a render context so we can
        rows, cols = self.vision_size
        width, height = cols, rows
        vision = self.sim.render(width, height, camera_name='vision', mode='offscreen')
        return np.array(vision, dtype='float32') / 255

    def obs_lidar(self, positions, group):
        '''
        Calculate and return a lidar observation.  See sub methods for implementation.
        '''
        if self.lidar_type == 'pseudo':
            return self.obs_lidar_pseudo(positions)
        elif self.lidar_type == 'natural':
            return self.obs_lidar_natural(group)
        else:
            raise ValueError(f'Invalid lidar_type {self.lidar_type}')

    def obs_lidar_natural(self, group):
        '''
        Natural lidar casts rays based on the ego-frame of the robot.
        Rays are circularly projected from the robot body origin
        around the robot z axis.
        '''
        body = self.model.body_name2id('robot')
        grp = np.asarray([i == group for i in range(int(const.NGROUP))], dtype='uint8')
        pos = np.asarray(self.world.robot_pos(), dtype='float64')
        mat_t = self.world.robot_mat()
        obs = np.zeros(self.lidar_num_bins)
        for i in range(self.lidar_num_bins):
            theta = (i / self.lidar_num_bins) * np.pi * 2
            vec = np.matmul(mat_t, theta2vec(theta))  # Rotate from ego to world frame
            vec = np.asarray(vec, dtype='float64')
            dist, _ = self.sim.ray_fast_group(pos, vec, grp, 1, body)
            if dist >= 0:
                obs[i] = np.exp(-dist)
        return obs

    def obs_lidar_pseudo(self, positions):
        '''
        Return a robot-centric lidar observation of a list of positions.

        Lidar is a set of bins around the robot (divided evenly in a circle).
        The detection directions are exclusive and exhaustive for a full 360 view.
        Each bin reads 0 if there are no objects in that direction.
        If there are multiple objects, the distance to the closest one is used.
        Otherwise the bin reads the fraction of the distance towards the robot.

        E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
        and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
        (The reading can be thought of as "closeness" or inverse distance)

        This encoding has some desirable properties:
            - bins read 0 when empty
            - bins smoothly increase as objects get close
            - maximum reading is 1.0 (where the object overlaps the robot)
            - close objects occlude far objects
            - constant size observation with variable numbers of objects
        '''
        obs = np.zeros(self.lidar_num_bins)
        for pos in positions:
            pos = np.asarray(pos)
            if pos.shape == (3,):
                pos = pos[:2]  # Truncate Z coordinate
            z = complex(*self.ego_xy(pos))  # X, Y as real, imaginary components
            dist = np.abs(z)
            angle = np.angle(z) % (np.pi * 2)
            bin_size = (np.pi * 2) / self.lidar_num_bins
            bin = int(angle / bin_size)
            bin_angle = bin_size * bin
            if self.lidar_max_dist is None:
                sensor = np.exp(-self.lidar_exp_gain * dist)
            else:
                sensor = max(0, self.lidar_max_dist - dist) / self.lidar_max_dist
            obs[bin] = max(obs[bin], sensor)
            # Aliasing
            if self.lidar_alias:
                alias = (angle - bin_angle) / bin_size
                assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
                bin_plus = (bin + 1) % self.lidar_num_bins
                bin_minus = (bin - 1) % self.lidar_num_bins
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        return obs

    def obs(self):
        ''' Return the observation of our agent '''
        self.sim.forward()  # Needed to get sensordata correct
        obs = {}

        if self.observe_goal_dist:
            obs['goal_dist'] = np.array([np.exp(-self.dist_goal())])
        if self.observe_goal_comp:
            obs['goal_compass'] = self.obs_compass(self.goal_pos)
        if self.observe_goal_lidar:
            obs['goal_lidar'] = self.obs_lidar([self.goal_pos], GROUP_GOAL)
        if self.task == 'push':
            box_pos = self.box_pos
            if self.observe_box_comp:
                obs['box_compass'] = self.obs_compass(box_pos)
            if self.observe_box_lidar:
                obs['box_lidar'] = self.obs_lidar([box_pos], GROUP_BOX)
        if self.task == 'circle' and self.observe_circle:
            obs['circle_lidar'] = self.obs_lidar([self.goal_pos], GROUP_CIRCLE)
        if self.observe_freejoint:
            joint_id = self.model.joint_name2id('robot')
            joint_qposadr = self.model.jnt_qposadr[joint_id]
            assert joint_qposadr == 0  # Needs to be the first entry in qpos
            obs['freejoint'] = self.data.qpos[:7]
        if self.observe_com:
            obs['com'] = self.world.robot_com()
        if self.observe_sensors:
            # Sensors which can be read directly, without processing
            for sensor in self.sensors_obs:  # Explicitly listed sensors
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.hinge_vel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.ballangvel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            # Process angular position sensors
            if self.sensors_angle_components:
                for sensor in self.robot.hinge_pos_names:
                    theta = float(self.world.get_sensor(sensor))  # Ensure not 1D, 1-element array
                    obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
                for sensor in self.robot.ballquat_names:
                    quat = self.world.get_sensor(sensor)
                    obs[sensor] = quat2mat(quat)
            else:  # Otherwise read sensors directly
                for sensor in self.robot.hinge_pos_names:
                    obs[sensor] = self.world.get_sensor(sensor)
                for sensor in self.robot.ballquat_names:
                    obs[sensor] = self.world.get_sensor(sensor)
        if self.observe_remaining:
            obs['remaining'] = np.array([self.steps / self.num_steps])
            assert 0.0 <= obs['remaining'][0] <= 1.0, 'bad remaining {}'.format(obs['remaining'])
        if self.walls_num and self.observe_walls:
            obs['walls_lidar'] = self.obs_lidar(self.walls_pos, GROUP_WALL)
        if self.observe_hazards:
            obs['hazards_lidar'] = self.obs_lidar(self.hazards_pos, GROUP_HAZARD)
        if self.observe_vases:
            obs['vases_lidar'] = self.obs_lidar(self.vases_pos, GROUP_VASE)
        if self.gremlins_num and self.observe_gremlins:
            obs['gremlins_lidar'] = self.obs_lidar(self.gremlins_obj_pos, GROUP_GREMLIN)
        if self.pillars_num and self.observe_pillars:
            obs['pillars_lidar'] = self.obs_lidar(self.pillars_pos, GROUP_PILLAR)
        if self.buttons_num and self.observe_buttons:
            # Buttons observation is zero while buttons are resetting
            if self.buttons_timer == 0:
                obs['buttons_lidar'] = self.obs_lidar(self.buttons_pos, GROUP_BUTTON)
            else:
                obs['buttons_lidar'] = np.zeros(self.lidar_num_bins)
        if self.observe_qpos:
            obs['qpos'] = self.data.qpos.copy()
        if self.observe_qvel:
            obs['qvel'] = self.data.qvel.copy()
        if self.observe_ctrl:
            obs['ctrl'] = self.data.ctrl.copy()
        if self.observe_vision:
            obs['vision'] = self.obs_vision()
        if self.observation_flatten:
            flat_obs = np.zeros(self.obs_flat_size)
            offset = 0
            for k in sorted(self.obs_space_dict.keys()):
                k_size = np.prod(obs[k].shape)
                flat_obs[offset:offset + k_size] = obs[k].flat
                offset += k_size
            obs = flat_obs

        obs = obs.astype(np.float32)
        assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'
        return obs


    def cost(self):
        ''' Calculate the current costs and return a dict '''
        self.sim.forward()  # Ensure positions and contacts are correct
        cost = {}
        # Conctacts processing
        if self.constrain_vases:
            cost['cost_vases_contact'] = 0
        if self.constrain_pillars:
            cost['cost_pillars'] = 0
        if self.constrain_buttons:
            cost['cost_buttons'] = 0
        if self.constrain_gremlins:
            cost['cost_gremlins'] = 0
        buttons_constraints_active = self.constrain_buttons and (self.buttons_timer == 0)
        for contact in self.data.contact[:self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom_id2name(g) for g in geom_ids])
            if self.constrain_vases and any(n.startswith('vase') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    cost['cost_vases_contact'] += self.vases_contact_cost
            if self.constrain_pillars and any(n.startswith('pillar') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    cost['cost_pillars'] += self.pillars_cost
            if buttons_constraints_active and any(n.startswith('button') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    if not any(n == f'button{self.goal_button}' for n in geom_names):
                        cost['cost_buttons'] += self.buttons_cost
            if self.constrain_gremlins and any(n.startswith('gremlin') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    cost['cost_gremlins'] += self.gremlins_contact_cost

        # Displacement processing
        if self.constrain_vases and self.vases_displace_cost:
            cost['cost_vases_displace'] = 0
            for i in range(self.num_objects_in_layout("vase", self.layout)):
                name = f'vase{i}'
                dist = np.sqrt(np.sum(np.square(self.data.get_body_xpos(name)[:2] - self.reset_layout[name])))
                if dist > self.vases_displace_threshold:
                    cost['cost_vases_displace'] += dist * self.vases_displace_cost

        # Velocity processing
        if self.constrain_vases and self.vases_velocity_cost:
            # TODO: penalize rotational velocity too, but requires another cost coefficient
            cost['cost_vases_velocity'] = 0
            for i in range(self.num_objects_in_layout("vase", self.layout)):
                name = f'vase{i}'
                vel = np.sqrt(np.sum(np.square(self.data.get_body_xvelp(name))))
                if vel >= self.vases_velocity_threshold:
                    cost['cost_vases_velocity'] += vel * self.vases_velocity_cost

        # Calculate constraint violations
        if self.constrain_hazards:
            cost['cost_hazards'] = 0
            for h_pos in self.hazards_pos:
                h_dist = self.dist_xy(h_pos)
                if h_dist <= self.hazards_size:
                    cost['cost_hazards'] += self.hazards_cost * (self.hazards_size - h_dist)

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        # Optionally remove shaping from reward functions.
        if self.constrain_indicator:
            for k in list(cost.keys()):
                cost[k] = float(cost[k] > 0.0)  # Indicator function

        self._cost = cost

        return cost

    def goal_met(self):
        ''' Return true if the current goal is met this step '''
        if self.task == 'goal':
            return self.dist_goal() <= self.goal_size
        if self.task == 'push':
            return self.dist_box_goal() <= self.goal_size
        if self.task == 'button':
            for contact in self.data.contact[:self.data.ncon]:
                geom_ids = [contact.geom1, contact.geom2]
                geom_names = sorted([self.model.geom_id2name(g) for g in geom_ids])
                if any(n == f'button{self.goal_button}' for n in geom_names):
                    if any(n in self.robot.geom_names for n in geom_names):
                        return True
            return False
        if self.task in ['x', 'z', 'circle', 'none', 'cleanup', 'cleanup_all']:
            return False
        raise ValueError(f'Invalid task {self.task}')

    def set_mocaps(self):
        ''' Set mocap object positions before a physics step is executed '''
        if self.gremlins_num: # self.constrain_gremlins:
            phase = float(self.data.time)
            for i in range(self.gremlins_num):
                name = f'gremlin{i}'
                target = np.array([np.sin(phase), np.cos(phase)]) * self.gremlins_travel
                pos = np.r_[target, [self.gremlins_size]]
                self.data.set_mocap_pos(name + 'mocap', pos)

    def update_layout(self):
        ''' Update layout dictionary with new places of objects '''
        self.sim.forward()
        for k in list(self.layout.keys()):
            # Mocap objects have to be handled separately
            if 'gremlin' in k:
                continue
            self.layout[k] = self.data.get_body_xpos(k)[:2].copy()

    def buttons_timer_tick(self):
        ''' Tick the buttons resampling timer '''
        self.buttons_timer = max(0, self.buttons_timer - 1)

    def step(self, action):
        ''' Take a step and return observation, reward, done, and info '''
        action = np.array(action, copy=False)  # Cast to ndarray
        assert not self.done, 'Environment must be reset before stepping'

        info = {}

        # Set action
        action_range = self.model.actuator_ctrlrange
        # action_scale = action_range[:,1] - action_range[:, 0]
        self.data.ctrl[:] = np.clip(action, action_range[:,0], action_range[:,1]) #np.clip(action * 2 / action_scale, -1, 1)
        if self.action_noise:
            self.data.ctrl[:] += self.action_noise * self.rs.randn(self.model.nu)

        # Simulate physics forward
        exception = False
        for _ in range(self.rs.binomial(self.frameskip_binom_n, self.frameskip_binom_p)):
            try:
                self.set_mocaps()
                self.sim.step()  # Physics simulation step
            except MujocoException as me:
                print('MujocoException', me)
                exception = True
                break
        if exception:
            self.done = True
            reward = self.reward_exception
            info['cost_exception'] = 1.0
        else:
            self.sim.forward()  # Needed to get sensor readings correct!

            # Reward processing
            reward = self.reward()

            # Constraint violations
            info.update(self.cost())

            # Button timer (used to delay button resampling)
            self.buttons_timer_tick()

            # Goal processing
            if self.goal_met():
                info['goal_met'] = True
                reward += self.reward_goal
                if self.continue_goal:
                    # Update the internal layout so we can correctly resample (given objects have moved)
                    self.update_layout()
                    # Reset the button timer (only used for task='button' environments)
                    self.buttons_timer = self.buttons_resampling_delay
                    # Try to build a new goal, end if we fail
                    if self.terminate_resample_failure:
                        try:
                            self.build_goal()
                        except ResamplingError as e:
                            # Normal end of episode
                            self.done = True
                    else:
                        # Try to make a goal, which could raise a ResamplingError exception
                        self.build_goal()
                else:
                    self.done = True

        # Timeout
        self.steps += 1
        if self.steps >= self.num_steps:
            self.done = True  # Maximum number of steps in an episode reached

        return self.obs(), reward, self.done, info

    def reward(self):
        ''' Calculate the dense component of reward.  Call exactly once per step '''
        reward = 0.0
        # Distance from robot to goal
        if self.task in ['goal', 'button']:
            dist_goal = self.dist_goal()
            reward += (self.last_dist_goal - dist_goal) * self.reward_distance
            self.last_dist_goal = dist_goal

        # distance of coloured blocks to respective goals
        if self.task == 'cleanup':
            dist_obj_goals = self.dist_obj_goals("sort")
            reward += np.sum((np.array(self.last_dist_obj_goals) - np.array(dist_obj_goals)) * self.reward_distance)
            self.last_dist_obj_goals = dist_obj_goals

            # proportion of blocks in goal
            task_completion = self.cleanup_task_completion("sort")
            reward += task_completion * self.reward_task_completion
            return reward

        # distance of coloured blocks to respective goals for all tasks
        if self.task == 'cleanup_all':
            reward = dict()
            for task in common.DOMAIN_TASK_IDS['cleanup_all']:
                dist_obj_goals = self.dist_obj_goals(task)
                task_reward = np.sum((np.array(self.last_dist_obj_goals[task]) - np.array(dist_obj_goals)) * self.reward_distance)
                self.last_dist_obj_goals[task] = dist_obj_goals

                # proportion of blocks in goal
                task_completion = self.cleanup_task_completion(task)
                task_reward += task_completion * self.reward_task_completion
                reward[task] = task_reward
            return reward

        # Distance from robot to box
        if self.task == 'push':
            dist_box = self.dist_box()
            gate_dist_box_reward = (self.last_dist_box > self.box_null_dist * self.box_size)
            reward += (self.last_dist_box - dist_box) * self.reward_box_dist * gate_dist_box_reward
            self.last_dist_box = dist_box
        # Distance from box to goal
        if self.task == 'push':
            dist_box_goal = self.dist_box_goal()
            reward += (self.last_box_goal - dist_box_goal) 
            self.last_box_goal = dist_box_goal
        # Used for forward locomotion tests
        if self.task == 'x':
            robot_com = self.world.robot_com()
            reward += (robot_com[0] - self.last_robot_com[0]) * self.reward_x
            self.last_robot_com = robot_com
        # Used for jump up tests
        if self.task == 'z':
            robot_com = self.world.robot_com()
            reward += (robot_com[2] - self.last_robot_com[2]) * self.reward_z
            self.last_robot_com = robot_com
        # Circle environment reward
        if self.task == 'circle':
            robot_com = self.world.robot_com()
            robot_vel = self.world.robot_vel()
            x, y, _ = robot_com
            u, v, _ = robot_vel
            radius = np.sqrt(x**2 + y**2)
            reward += (((-u*y + v*x)/radius)/(1 + np.abs(radius - self.circle_radius))) * self.reward_circle
        # Intrinsic reward for uprightness
        if self.reward_orientation:
            zalign = quat2zalign(self.data.get_body_xquat(self.reward_orientation_body))
            reward += self.reward_orientation_scale * zalign
        # Clip reward
        if self.reward_clip:
            in_range = reward < self.reward_clip and reward > -self.reward_clip
            if not(in_range):
                reward = np.clip(reward, -self.reward_clip, self.reward_clip)
                print('Warning: reward was outside of range!')
        return reward

    def render_lidar(self, poses, color, offset, group):
        ''' Render the lidar observation '''
        robot_pos = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        lidar = self.obs_lidar(poses, group)
        for i, sensor in enumerate(lidar):
            if self.lidar_type == 'pseudo':
                i += 0.5  # Offset to center of bin
            theta = 2 * np.pi * i / self.lidar_num_bins
            rad = self.render_lidar_radius
            binpos = np.array([np.cos(theta) * rad, np.sin(theta) * rad, offset])
            pos = robot_pos + np.matmul(binpos, robot_mat.transpose())
            alpha = min(1, sensor + .1)
            self.viewer.add_marker(pos=pos,
                                   size=self.render_lidar_size * np.ones(3),
                                   type=const.GEOM_SPHERE,
                                   rgba=np.array(color) * alpha,
                                   label='')

    def render_compass(self, pose, color, offset):
        ''' Render a compass observation '''
        robot_pos = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        # Truncate the compass to only visualize XY component
        compass = np.concatenate([self.obs_compass(pose)[:2] * 0.15, [offset]])
        pos = robot_pos + np.matmul(compass, robot_mat.transpose())
        self.viewer.add_marker(pos=pos,
                               size=.05 * np.ones(3),
                               type=const.GEOM_SPHERE,
                               rgba=np.array(color) * 0.5,
                               label='')

    def render_area(self, pos, size, color, label='', alpha=0.1):
        ''' Render a radial area in the environment '''
        z_size = min(size, 0.3)
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(pos=pos,
                               size=[size, size, z_size],
                               type=const.GEOM_CYLINDER,
                               rgba=np.array(color) * alpha,
                               label=label if self.render_labels else '')

    def render_sphere(self, pos, size, color, label='', alpha=0.1):
        ''' Render a radial area in the environment '''
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(pos=pos,
                               size=size * np.ones(3),
                               type=const.GEOM_SPHERE,
                               rgba=np.array(color) * alpha,
                               label=label if self.render_labels else '')

    def render_swap_callback(self):
        ''' Callback between mujoco render and swapping GL buffers '''
        if self.observe_vision and self.vision_render:
            self.viewer.draw_pixels(self.save_obs_vision, 0, 0)

    def render(self,
               mode='human', 
               camera_id=None,
               width=DEFAULT_WIDTH,
               height=DEFAULT_HEIGHT
               ):
        ''' Render the environment to the screen '''

        if self.viewer is None or mode!=self._old_render_mode:
            # Set camera if specified
            if mode == 'human':
                self.viewer = MjViewer(self.sim)
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.type = const.CAMERA_FREE
            else:
                self.viewer = MjRenderContextOffscreen(self.sim)
                self.viewer._hide_overlay = True
                self.viewer.cam.fixedcamid = camera_id #self.model.camera_name2id(mode)
                self.viewer.cam.type = const.CAMERA_FIXED
            self.viewer.render_swap_callback = self.render_swap_callback
            # Turn all the geom groups on
            self.viewer.vopt.geomgroup[:] = 1
            self._old_render_mode = mode
        self.viewer.update_sim(self.sim)

        if camera_id is not None:
            # Update camera if desired
            self.viewer.cam.fixedcamid = camera_id

        # Lidar markers
        if self.render_lidar_markers:
            offset = self.render_lidar_offset_init  # Height offset for successive lidar indicators
            if 'box_lidar' in self.obs_space_dict or 'box_compass' in self.obs_space_dict:
                if 'box_lidar' in self.obs_space_dict:
                    self.render_lidar([self.box_pos], COLOR_BOX, offset, GROUP_BOX)
                if 'box_compass' in self.obs_space_dict:
                    self.render_compass(self.box_pos, COLOR_BOX, offset)
                offset += self.render_lidar_offset_delta
            if 'goal_lidar' in self.obs_space_dict or 'goal_compass' in self.obs_space_dict:
                if 'goal_lidar' in self.obs_space_dict:
                    self.render_lidar([self.goal_pos], COLOR_GOAL, offset, GROUP_GOAL)
                if 'goal_compass' in self.obs_space_dict:
                    self.render_compass(self.goal_pos, COLOR_GOAL, offset)
                offset += self.render_lidar_offset_delta
            if 'buttons_lidar' in self.obs_space_dict:
                self.render_lidar(self.buttons_pos, COLOR_BUTTON, offset, GROUP_BUTTON)
                offset += self.render_lidar_offset_delta
            if 'circle_lidar' in self.obs_space_dict:
                self.render_lidar([ORIGIN_COORDINATES], COLOR_CIRCLE, offset, GROUP_CIRCLE)
                offset += self.render_lidar_offset_delta
            if 'walls_lidar' in self.obs_space_dict:
                self.render_lidar(self.walls_pos, COLOR_WALL, offset, GROUP_WALL)
                offset += self.render_lidar_offset_delta
            if 'hazards_lidar' in self.obs_space_dict:
                self.render_lidar(self.hazards_pos, COLOR_HAZARD, offset, GROUP_HAZARD)
                offset += self.render_lidar_offset_delta
            if 'pillars_lidar' in self.obs_space_dict:
                self.render_lidar(self.pillars_pos, COLOR_PILLAR, offset, GROUP_PILLAR)
                offset += self.render_lidar_offset_delta
            if 'gremlins_lidar' in self.obs_space_dict:
                self.render_lidar(self.gremlins_obj_pos, COLOR_GREMLIN, offset, GROUP_GREMLIN)
                offset += self.render_lidar_offset_delta
            if 'vases_lidar' in self.obs_space_dict:
                self.render_lidar(self.vases_pos, COLOR_VASE, offset, GROUP_VASE)
                offset += self.render_lidar_offset_delta

        # Add goal marker
        if self.task == 'button':
            self.render_area(self.goal_pos, self.buttons_size * 2, COLOR_BUTTON, 'goal', alpha=0.1)

        # Add indicator for nonzero cost
        # if self._cost.get('cost', 0) > 0:
        #     self.render_sphere(self.world.robot_pos(), 0.25, COLOR_RED, alpha=.5)

        # Draw vision pixels
        if self.observe_vision and self.vision_render:
            vision = self.obs_vision()
            vision = np.array(vision * 255, dtype='uint8')
            vision = Image.fromarray(vision).resize(self.vision_render_size)
            vision = np.array(vision, dtype='uint8')
            self.save_obs_vision = vision

        if mode=='human':
            self.viewer.render()
            data = self.viewer.read_pixels(width, height, depth=False)
            self.viewer._markers[:] = []
            self.viewer._overlay.clear()
            return data[::-1, :, :]
        elif mode=='rgb_array':
            self.viewer.render(width, height)
            data = self.viewer.read_pixels(width, height, depth=False)
            self.viewer._markers[:] = []
            self.viewer._overlay.clear()
            return data[::-1, :, :]