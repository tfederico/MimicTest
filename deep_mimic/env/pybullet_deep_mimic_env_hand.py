import numpy as np
import math
from deep_mimic.env.env import Env
from deep_mimic.env.action_space import ActionSpace
from pybullet_utils import bullet_client
import time
from deep_mimic.env import motion_capture_data
from deep_mimic.env import simple_hand_stable_pd as hand_stable_pd
import data as pybullet_data
import pybullet as p1
import random

from enum import Enum

class InitializationStrategy(Enum):
    """Set how the environment is initialized."""
    START = 0
    RANDOM = 1  # random state initialization (RSI)


class PyBulletDeepMimicEnv(Env):

    def __init__(self, arg_parser=None, enable_draw=False, pybullet_client=None,
                 time_step=1./240,
                 init_strategy=InitializationStrategy.RANDOM,
                 use_com_reward=False):
        super().__init__(arg_parser, enable_draw)
        self._num_agents = 1
        self._pybullet_client = pybullet_client
        self._isInitialized = False
        self._arg_parser = arg_parser
        self.timeStep = time_step
        self._init_strategy = init_strategy
        print("Initialization strategy: {:s}".format(init_strategy))
        self.reset()

    def reset(self):

        if not self._isInitialized:
            if self.enable_draw:
                self._pybullet_client = bullet_client.BulletClient(connection_mode=p1.GUI)
                #disable 'GUI' since it slows down a lot on Mac OSX and some other platforms
                self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI, 0)
            else:
                self._pybullet_client = bullet_client.BulletClient()

            self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
            z2y = self._pybullet_client.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
            self._planeId = self._pybullet_client.loadURDF("plane_implicit.urdf", [0, 0, 0],
                                                           z2y,
                                                           useMaximalCoordinates=True)

            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
            self._pybullet_client.setGravity(0, 0, 0)

            self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
            self._pybullet_client.changeDynamics(self._planeId, linkIndex=-1, lateralFriction=0.9)

            self._mocapData = motion_capture_data.MotionCaptureData()

            motion_file = self._arg_parser.parse_strings('motion_file')
            print("motion_file=", motion_file[0])
            motionPath = pybullet_data.getDataPath() + "/" + motion_file[0]
            print(motionPath)
            self._mocapData.Load(motionPath)
            timeStep = self.timeStep
            useFixedBase = True
            self._humanoid = hand_stable_pd.HandStablePD(self._pybullet_client, self._mocapData,
                                                     timeStep, useFixedBase, self._arg_parser)
            self._isInitialized = True

            self._pybullet_client.setTimeStep(timeStep)
            self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

        if self._init_strategy == InitializationStrategy.RANDOM:
            rnrange = 1000
            rn = random.randint(0, rnrange)
            startTime = float(rn) / rnrange * self._humanoid.getCycleTime()
        elif self._init_strategy == InitializationStrategy.START:
            startTime = 0

        self.t = startTime

        self._humanoid.setSimTime(startTime)

        self._humanoid.resetPose()
        #this clears the contact points. Todo: add API to explicitly clear all contact points?
        #self._pybullet_client.stepSimulation()
        self._humanoid.resetPose()
        self.needs_update_time = self.t - 1  #force update

    def get_num_agents(self):
        return self._num_agents

    def get_action_space(self, agent_id):
        return ActionSpace(ActionSpace.Continuous)

    def get_reward_min(self, agent_id):
        return 0

    def get_reward_max(self, agent_id):
        return 1

    def get_reward_fail(self, agent_id):
        return self.get_reward_min(agent_id)

    def get_reward_succ(self, agent_id):
        return self.get_reward_max(agent_id)

    def get_state_size(self, agent_id):
        #cCtController::GetStateSize()
        #int state_size = cDeepMimicCharController::GetStateSize();
        #                     state_size += GetStatePoseSize();#(4+3)*16 + 1 = 113
        #                     state_size += GetStateVelSize(); #(4+3-1)*16 = 96
        #state_size += GetStatePhaseSize();#1 TODO: check how this number is calculated
        # 210
        return 210

    def build_state_norm_groups(self, agent_id):
        groups = [0] * self.get_state_size(agent_id)
        groups[0] = -1
        return groups

    def build_state_offset(self, agent_id):
        out_offset = [0] * self.get_state_size(agent_id)
        phase_offset = -0.5
        out_offset[0] = phase_offset
        return np.array(out_offset)

    def build_state_scale(self, agent_id):
        out_scale = [1] * self.get_state_size(agent_id)
        phase_scale = 2
        out_scale[0] = phase_scale
        return np.array(out_scale)

    def get_goal_size(self, agent_id):
        return 0

    def get_action_size(self, agent_id):
        ctrl_size = 22  #numDof
        root_size = 7
        return ctrl_size - root_size

    def build_goal_norm_groups(self, agent_id):
        return np.array([])

    def build_goal_offset(self, agent_id):
        return np.array([])

    def build_goal_scale(self, agent_id):
        return np.array([])

    def build_action_offset(self, agent_id):
        out_offset = [0] * self.get_action_size(agent_id)
        out_offset = [
            -0.5 * (up + low) for up, low in zip(self.build_action_bound_max(-1), self.build_action_bound_min(-1))
        ]
        #see cCtCtrlUtil::BuildOffsetScalePDPrismatic and
        #see cCtCtrlUtil::BuildOffsetScalePDSpherical
        return np.array(out_offset)

    def build_action_scale(self, agent_id):
        out_scale = [1] * self.get_action_size(agent_id)
        #see cCtCtrlUtil::BuildOffsetScalePDPrismatic and
        #see cCtCtrlUtil::BuildOffsetScalePDSpherical
        out_scale = [
            0.5/(up - low) for up, low in zip(self.build_action_bound_max(-1), self.build_action_bound_min(-1))
        ]
        return np.array(out_scale)

    def build_action_bound_min(self, agent_id):
        #see cCtCtrlUtil::BuildBoundsPDSpherical
        out_scale = [-1] * self.get_action_size(agent_id)
        out_scale = [
            0
        ] * self.get_action_size(-1)

        return out_scale

    def build_action_bound_max(self, agent_id):
        out_scale = [1] * self.get_action_size(agent_id)
        out_scale = [
            1.57
        ] * self.get_action_size(-1)
        return out_scale

    def set_mode(self, mode):
        self._mode = mode

    def record_state(self, agent_id):
        state = self._humanoid.getState()

        return np.array(state)

    def record_goal(self, agent_id):
        return np.array([])

    def calc_reward(self, agent_id):
        kinPose = self._humanoid.computePose(self._humanoid._frameFraction)
        reward = self._humanoid.getReward(kinPose)
        return reward

    def set_action(self, agent_id, action):
        self.desiredPose = self._humanoid.convertActionToPose(action)
        #we need the target root positon and orientation to be zero, to be compatible with deep mimic
        self.desiredPose[0] = 0
        self.desiredPose[1] = 0
        self.desiredPose[2] = 0
        self.desiredPose[3] = 0
        self.desiredPose[4] = 0
        self.desiredPose[5] = 0
        self.desiredPose[6] = 0

    def log_val(self, agent_id, val):
        pass

    def update(self, timeStep):
        self._pybullet_client.setTimeStep(timeStep)
        self._humanoid._timeStep = timeStep
        self.timeStep = timeStep


        self.t += timeStep
        self._humanoid.setSimTime(self.t)

        if self.desiredPose:
            kinPose = self._humanoid.computePose(self._humanoid._frameFraction)
            self._humanoid.initializePose(self._humanoid._poseInterpolator,
                                      self._humanoid._kin_model,
                                      initBase=True)

            maxForces = [
                0, 0, 0,
                0, 0, 0, 0
            ] + [100] * 15


            self._humanoid.computeAndApplyPDForces(self.desiredPose,
                                                   maxForces=maxForces)

            self._pybullet_client.stepSimulation()

    def set_sample_count(self, count):
        return

    def check_terminate(self, agent_id):
        return Env.Terminate(self.is_episode_end())

    def is_episode_end(self):
        isEnded = self._humanoid.terminates()
        #also check maximum time, 20 seconds (todo get from file)
        #print("self.t=",self.t)
        if (self.t > 20):
            isEnded = True
        return isEnded

    def check_valid_episode(self):
        #could check if limbs exceed velocity threshold
        return True

    def getKeyboardEvents(self):
        return self._pybullet_client.getKeyboardEvents()

    def isKeyTriggered(self, keys, key):
        o = ord(key)
        #print("ord=",o)
        if o in keys:
            return keys[ord(key)] & self._pybullet_client.KEY_WAS_TRIGGERED
        return False
