import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import data as pybullet_data
from deep_mimic.env.pybullet_deep_mimic_env_whole import PyBulletDeepMimicEnv, InitializationStrategy
from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger
from scipy.stats import truncnorm
import wandb

logger = logging.getLogger(__name__)

class CheatingBox(gym.spaces.Box):
    def __init__(self, low, high, shape=None, dtype=np.float32):
        super().__init__(low, high, shape, dtype)

    def sample(self):
        return truncnorm.rvs(self.low, self.high, size=self.shape[0])


class WholeDeepBulletEnv(gym.Env):
    """Base Gym environment for the DeepMimic motion imitation tasks."""
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}



    def __init__(self, renders=False, arg_file='', test_mode=False,
                 time_step=1./240,
                 rescale_actions=True,
                 rescale_observations=True,
                 use_com_reward=False):
        """Instantiate a DeepMimic motion imitation environment.

        Args:
          test_mode (bool): in test mode, the `reset()` method will always set the mocap clip time to 0
            at the beginning of every episode.
          time_step (float): physics time step.
          rescale_actions (bool): rescale the actions using the bounds on the action space.
          rescale_observations (bool): rescale the observations using the bounds on the observation space.
          use_com_reward (bool): whether to use the center-of-mass reward.
        """
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
        self.internal_env = None
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
        self._log_freq = 100
        self.test_mode = test_mode
        if self.test_mode:
            print("Environment running in TEST mode")

        # cam options
        self._cam_dist = 3
        self._cam_pitch = -30
        self._cam_yaw = -60
        self._cam_roll = 0

        self.reset()

        # Query the policy at 30Hz
        self.policy_query_30 = False
        if self.policy_query_30:
            self._policy_step = 1./30
        else:
            self._policy_step = 1./240
        self._num_env_steps = int(self._policy_step / self._time_step)

        action_bound_min = np.array(self.internal_env.build_action_bound_min(-1))
        action_bound_max = np.array(self.internal_env.build_action_bound_max(-1))

        if self._rescale_actions:
            action_bound_min = self.scale_action(action_bound_min)
            action_bound_max = self.scale_action(action_bound_max)

        self.action_space = CheatingBox(action_bound_min, action_bound_max)

        observation_min = np.array([0.0]+[-100.0]+[-4.0]*273+[-500.0]*234)
        observation_max = np.array([1.0]+[100.0]+[4.0]*273+[500.0]*234)

        if self._rescale_observations:
            observation_min = self.scale_observation(observation_min)
            observation_max = self.scale_observation(observation_max)

        state_size = self.internal_env.get_state_size(-1)
        self.observation_space = spaces.Box(observation_min, observation_max, dtype=np.float32)

        self.seed()

        self.viewer = None
        self._configure()

    def scale_action(self, action):
        mean = -self._action_offset
        std = 1./self._action_scale
        return (action - mean) / (std)

    def scale_observation(self, state):
        mean = -self._state_offset
        std = 1./self._state_scale
        return (state - mean) / (std)

    def unscale_observation(self, scaled_state):
        mean = -self._state_offset
        std = 1./self._state_scale
        return scaled_state * std + mean

    def unscale_action(self, scaled_action):
        mean = -self._action_offset
        std = 1./self._action_scale
        return scaled_action * std + mean

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        agent_id = self.agent_id

        if self._rescale_actions:
            # Rescale the action
            action = self.unscale_action(action)

        # Record reward
        reward = self.internal_env.calc_reward(agent_id)
        assert reward <= 1, "Reward is supposed to be <= 1. Actual reward: {}".format(reward)
        assert reward >= 0, "Reward is supposed to be >= 0. Actual reward: {}".format(reward)

        # Apply control action
        self.internal_env.set_action(agent_id, action)

        start_time = self.internal_env.t

        # step sim
        for i in range(self._num_env_steps):
            self.internal_env.update(self._time_step)

        elapsed_time = self.internal_env.t - start_time

        self._numSteps += 1

        # Record state
        self.state = self.internal_env.record_state(agent_id)
        state = self.state

        if self._rescale_observations:
            state = self.scale_observation(state)

        # Record done
        done = self.internal_env.is_episode_end()

        self.camera_update()

        # get the reward info
        info = {
            'reward': self.internal_env._humanoid._info_rew,
            'error': self.internal_env._humanoid._info_err
        }

        return state, reward, done, info

    def reset(self):
        # use the initialization strategy
        if self.internal_env is None:
            if self.test_mode:
                init_strat = InitializationStrategy.START
            else:
                init_strat = InitializationStrategy.RANDOM
            self.internal_env = PyBulletDeepMimicEnv(self._arg_parser, self._renders,
                                                     time_step=self._time_step,
                                                     init_strategy=init_strat,
                                                     use_com_reward=self._use_com_reward)

        self.internal_env.reset()
        self._p = self.internal_env._pybullet_client
        agent_id = self.agent_id  # unused here
        self._state_offset = self.internal_env.build_state_offset(self.agent_id)
        self._state_scale = self.internal_env.build_state_scale(self.agent_id)
        self._action_offset = self.internal_env.build_action_offset(self.agent_id)
        self._action_scale = self.internal_env.build_action_scale(self.agent_id)
        self._numSteps = 0
        # Record state
        self.state = self.internal_env.record_state(agent_id)

        self.camera_update()

        # return state as ndarray
        state = np.array(self.state)
        if self._rescale_observations:
            state = self.scale_observation(state)
        return state



    def render(self, mode='human', close=False):
        if mode == "human":
            self._renders = True
        if mode != "rgb_array":
            return np.array([])
        hand = self.internal_env._humanoid
        base_pos, orn = self._p.getBasePositionAndOrientation(hand._sim_model)
        base_pos = np.asarray(base_pos)
        # track the position
        base_pos[1] += 0.3
        rpy = self._p.getEulerFromQuaternion(orn)  # rpy, in radians
        rpy = 180 / np.pi * np.asarray(rpy)  # convert rpy in degrees

        if (not self._p == None):
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=self._cam_roll,
                upAxisIndex=1)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(self._render_width) / self._render_height,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
        else:
            px = np.array([[[255,255,255,255]]*self._render_width]*self._render_height, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def configure(self, args):
        pass

    def close(self):
        pass

    def camera_update(self):
        """Update the debug visualizer camera."""
        # update camera
        hand = self.internal_env._humanoid
        base_pos, base_orn = self._p.getBasePositionAndOrientation(
            hand._sim_model)
        debug_caminfo = self._p.getDebugVisualizerCamera()
        (yaw, pitch, cur_dist) = debug_caminfo[8:11]
        # self._cam_dist = cur_dist
        self._p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=base_pos)

class WholeDeepMimicSignerBulletEnv(WholeDeepBulletEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False, arg_file="run_humanoid3d_tuning_motion_whole_args.txt", test_mode=False):
        # start the bullet physics server
        WholeDeepBulletEnv.__init__(self, renders, arg_file, test_mode=test_mode)
