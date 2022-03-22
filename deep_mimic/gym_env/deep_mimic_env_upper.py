"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""
# import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import data as pybullet_data
from deep_mimic.env.pybullet_deep_mimic_env_upper import PyBulletDeepMimicEnvUpper, InitializationStrategy
from deep_mimic.gym_env.deep_mimic_env_selector import HumanoidDeepBulletSelectorEnv
from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger
from scipy.stats import truncnorm


logger = logging.getLogger(__name__)

class CheatingBox(gym.spaces.Box):
    def __init__(self, low, high, shape=None, dtype=np.float32):
        super().__init__(low, high, shape, dtype)

    def sample(self):
        return truncnorm.rvs(self.low, self.high, size=self.shape[0])


class HumanoidDeepBulletUpperEnv(HumanoidDeepBulletSelectorEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}



    def __init__(self, renders=False, arg_file='', test_mode=False,
                 time_step=1./240,
                 rescale_actions=True,
                 rescale_observations=True,
                 use_com_reward=False):
        #super().__init__(renders, arg_file, test_mode, time_step, rescale_actions, rescale_observations, use_com_reward)
        self._arg_parser = ArgParser()
        Logger.print2("===========================================================")
        succ = False
        if (arg_file != ''):
            path = pybullet_data.getDataPath() + "/args/" + arg_file
            succ = self._arg_parser.load_file(path)
            Logger.print2(arg_file)
        assert succ, Logger.print2('Failed to load args from: ' + arg_file)

        self._p = None
        self._time_step = time_step
        self._internal_env = None
        self._renders = renders
        self._discrete_actions = False
        self._arg_file = arg_file
        self._render_height = 480
        self._render_width = 854
        self._rescale_actions = rescale_actions
        self._rescale_observations = rescale_observations
        self._use_com_reward = use_com_reward
        self.agent_id = -1

        self._numSteps = None
        self.test_mode = test_mode
        if self.test_mode:
            print("Environment running in TEST mode")

        # cam options
        self._cam_dist = 3
        self._cam_pitch = 0.3
        self._cam_yaw = 0.1
        self._cam_roll = 0

        self.reset()

        # Query the policy at 30Hz
        self.policy_query_30 = True
        if self.policy_query_30:
            self._policy_step = 1./30
        else:
            self._policy_step = 1./240
        self._num_env_steps = int(self._policy_step / self._time_step)

        ctrl_size = 43 - (4 + 4 + 1)*2  #numDof
        root_size = 7 # root

        action_dim = ctrl_size - root_size

        #print("len(action_bound_min)=",len(action_bound_min))
        action_bound_min = np.array(self._internal_env.build_action_bound_min(-1))
        #print("len(action_bound_max)=",len(action_bound_max))
        action_bound_max = np.array(self._internal_env.build_action_bound_max(-1))

        if self._rescale_actions:
            action_bound_min = self.scale_action(action_bound_min)
            action_bound_max = self.scale_action(action_bound_max)

        self.action_space = CheatingBox(action_bound_min, action_bound_max)#spaces.Box(action_bound_min, action_bound_max)

        self.n_clips = self._internal_env.get_num_clips()
        clips_min = []
        clips_max = []
        if self.n_clips > 1:
            clips_min = [0.0] * self.n_clips
            clips_max = [1.0] * self.n_clips
        observation_min = np.array([0.0]+[-100.0]+[-4.0]*63+[-500.0]*54+clips_min)
        observation_max = np.array([1.0]+[100.0]+[4.0]*63+[500.0]*54+clips_max)

        if self._rescale_observations:
            observation_min = self.scale_observation(observation_min)
            observation_max = self.scale_observation(observation_max)

        state_size = self._internal_env.get_state_size(-1)
        self.observation_space = spaces.Box(observation_min, observation_max, dtype=np.float32)
        self.switch_steps = 300
        self.seed()

        self.viewer = None
        self._configure()

    def reset(self):
        # use the initialization strategy
        if self._internal_env is None:
            if self.test_mode:
                init_strat = InitializationStrategy.START
            else:
                init_strat = InitializationStrategy.RANDOM
            self._internal_env = PyBulletDeepMimicEnvUpper(self._arg_parser, self._renders,
                                                      time_step=self._time_step,
                                                      init_strategy=init_strat,
                                                      use_com_reward=self._use_com_reward)
        #self.internal_env.change_current_clip()
        self._internal_env.reset()
        #print("B", self.internal_env.get_current_clip_num())
        self._p = self._internal_env._pybullet_client
        agent_id = self.agent_id  # unused here
        self._state_offset = self._internal_env.build_state_offset(self.agent_id)
        self._state_scale = self._internal_env.build_state_scale(self.agent_id)
        self._action_offset = self._internal_env.build_action_offset(self.agent_id)
        self._action_scale = self._internal_env.build_action_scale(self.agent_id)
        self._numSteps = 0
        # Record state
        self.state = self._internal_env.record_state(agent_id)

        self.camera_update()

        # return state as ndarray
        state = np.array(self.state)
        if self._rescale_observations:
            state = self.scale_observation(state)
        return state

class HumanoidDeepMimicUpperSignerBulletEnv(HumanoidDeepBulletUpperEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False):
        # start the bullet physics server
        HumanoidDeepBulletUpperEnv.__init__(self, renders, arg_file="run_humanoid3d_signer_args.txt")