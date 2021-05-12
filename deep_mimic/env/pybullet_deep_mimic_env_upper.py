import numpy as np
import math
from deep_mimic.env.pybullet_deep_mimic_env_multiclip import InitializationStrategy
from deep_mimic.env.pybullet_deep_mimic_env_selector import PyBulletDeepMimicEnvSelector
from pybullet_utils import bullet_client
import time
from deep_mimic.env import motion_capture_data_multiclip
from deep_mimic.env import humanoid_stable_pd_upper
import data as pybullet_data
import pybullet as p1
import random



class PyBulletDeepMimicEnvUpper(PyBulletDeepMimicEnvSelector):

    def __init__(self, arg_parser=None, enable_draw=False, pybullet_client=None,
                 time_step=1. / 240,
                 init_strategy=InitializationStrategy.RANDOM,
                 use_com_reward = False):
        #super().__init__(arg_parser, enable_draw, pybullet_clinet, time_step, init_strategy, use_com_reward)
        self._num_agents = 1
        self._pybullet_client = pybullet_client
        self._isInitialized = False
        #self._useStablePD = True
        self._arg_parser = arg_parser
        self.timeStep = time_step
        self._init_strategy = init_strategy
        self._n_clips = -1
        self._current_clip = -1
        self._use_com_reward = use_com_reward
        print("Initialization strategy: {:s}".format(init_strategy))
        self.enable_draw = enable_draw
        self.reset()

    def reset(self):

        if not self._isInitialized:
            if self.enable_draw:
                self._pybullet_client = bullet_client.BulletClient(connection_mode=p1.GUI)
                # disable 'GUI' since it slows down a lot on Mac OSX and some other platforms
                self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI, 0)
            else:
                self._pybullet_client = bullet_client.BulletClient()

            self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
            z2y = self._pybullet_client.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
            self._planeId = self._pybullet_client.loadURDF("plane_implicit.urdf", [0, 0, 0],
                                                           z2y,
                                                           useMaximalCoordinates=True)
            # print("planeId=",self._planeId)
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
            self._pybullet_client.setGravity(0, -9.8, 0)

            self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
            self._pybullet_client.changeDynamics(self._planeId, linkIndex=-1, lateralFriction=0.9)

            self._mocapData = motion_capture_data_multiclip.MotionCaptureDataMultiClip()

            motion_file = self._arg_parser.parse_strings('motion_file')
            print("motion_file=", motion_file[0])

            motionPath = pybullet_data.getDataPath() + "/" + motion_file[0]

            self._mocapData.Load(motionPath)
            self._n_clips = self._mocapData.getNumClips()
            #assert self._n_clips > 1, "You should use at least two clips"
            self._current_clip = random.randint(0, self._n_clips - 1)
            if self._n_clips > 1:
                self._one_hot_current_clip = np.eye(self._n_clips)[self._current_clip]
            timeStep = self.timeStep
            useFixedBase = False
            self._humanoid = humanoid_stable_pd_upper.HumanoidStablePDUpper(self._pybullet_client, self._mocapData,
                                                                                  timeStep, useFixedBase, self._arg_parser,
                                                                                  self._use_com_reward, self._current_clip)
            self._isInitialized = True

            self._pybullet_client.setTimeStep(timeStep)
            self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

        # print("numframes = ", self._humanoid._mocap_data.NumFrames())

        if self._init_strategy == InitializationStrategy.RANDOM:
            rnrange = 1000
            rn = random.randint(0, rnrange)
            startTime = float(rn) / rnrange * self._humanoid.getCycleTime(self._current_clip)
        elif self._init_strategy == InitializationStrategy.START:
            startTime = 0

        self.t = startTime
        self.remaining_t = 20.0
        self._humanoid.setSimTime(startTime)

        self._humanoid.resetPose()
        # this clears the contact points. Todo: add API to explicitly clear all contact points?
        self._humanoid.resetPose()
        self.needs_update_time = self.t - 1  # force update

    # scene_name == "imitate" -> cDrawSceneImitate
    def get_state_size(self, agent_id):
        # cCtController::GetStateSize()
        # int state_size = cDeepMimicCharController::GetStateSize();
        #                     state_size += GetStatePoseSize();#64
        #                     state_size += GetStateVelSize(); #(3+3)*numBodyParts=54
        # state_size += GetStatePhaseSize();#1
        # 197
        if self._n_clips > 1:
            return 119 + self._n_clips
        return 119

    def get_action_size(self, agent_id):
        ctrl_size = 43 - 9*2  # numDof
        root_size = 7
        return ctrl_size - root_size


    def build_action_offset(self, agent_id):
        out_offset = [
            0.0000000000, 0.0000000000, 0.0000000000, -0.200000000,
            0.0000000000, 0.0000000000, 0.0000000000, -0.200000000,
            #0.0000000000, 0.0000000000, 0.00000000, -0.2000000,
            #1.57000000,
            #0.00000000, 0.00000000, 0.00000000, -0.2000000,
            0.00000000, 0.00000000, 0.00000000, -0.2000000,
            -1.5700000,
            #0.00000000, 0.00000000, 0.00000000, -0.2000000,
            #1.57000000,
            #0.00000000, 0.00000000, 0.00000000, -0.2000000,
            0.00000000, 0.00000000, 0.00000000, -0.2000000,
            -1.5700000
        ]
        # see cCtCtrlUtil::BuildOffsetScalePDPrismatic and
        # see cCtCtrlUtil::BuildOffsetScalePDSpherical
        return np.array(out_offset)

    def build_action_scale(self, agent_id):
        # see cCtCtrlUtil::BuildOffsetScalePDPrismatic and
        # see cCtCtrlUtil::BuildOffsetScalePDSpherical
        out_scale = [
            0.20833333333333, 1.00000000000000, 1.00000000000000, 1.00000000000000,
            0.25000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
            #0.12077294685990, 1.00000000000000, 1.000000000000, 1.000000000000,
            #0.159235668789,
            #0.159235668789, 1.000000000000, 1.000000000000, 1.000000000000,
            0.079617834394, 1.000000000000, 1.000000000000, 1.000000000000,
            0.159235668789,
            #0.120772946859, 1.000000000000, 1.000000000000, 1.000000000000,
            #0.159235668789,
            #0.159235668789, 1.000000000000, 1.000000000000, 1.000000000000,
            0.107758620689, 1.000000000000, 1.000000000000, 1.000000000000,
            0.159235668789
        ]
        return np.array(out_scale)

    def build_action_bound_min(self, agent_id):
        # see cCtCtrlUtil::BuildBoundsPDSpherical
        out_scale = [
            -4.79999999999, -1.00000000000, -1.00000000000, -1.00000000000,
            -4.00000000000, -1.00000000000, -1.00000000000, -1.00000000000,
            #-7.77999999999, -1.00000000000, -1.000000000, -1.000000000,
            #-7.850000000,
            #-6.280000000, -1.000000000, -1.000000000, -1.000000000,
            -12.56000000, -1.000000000, -1.000000000, -1.000000000,
            -4.710000000,
            #-7.779999999, -1.000000000, -1.000000000, -1.000000000,
            #-7.850000000,
            #-6.280000000, -1.000000000, -1.000000000, -1.000000000,
            -8.460000000, -1.000000000, -1.000000000, -1.000000000,
            -4.710000000
        ]

        return out_scale

    def build_action_bound_max(self, agent_id):
        out_scale = [
            4.799999999, 1.000000000, 1.000000000, 1.000000000,
            4.000000000, 1.000000000, 1.000000000, 1.000000000,
            #8.779999999, 1.000000000, 1.0000000, 1.0000000,
            #4.7100000,
            #6.2800000, 1.0000000, 1.0000000, 1.0000000,
            12.560000, 1.0000000, 1.0000000, 1.0000000,
            7.8500000,
            #8.7799999, 1.0000000, 1.0000000, 1.0000000,
            #4.7100000,
            #6.2800000, 1.0000000, 1.0000000, 1.0000000,
            10.100000, 1.0000000, 1.0000000, 1.0000000,
            7.8500000
        ]
        return out_scale

    def record_state(self, agent_id):
        state = self._humanoid.getState()
        phase = [state[0]]
        root_y = [state[1]]
        rots = state[2:23] + state[44:65] + state[86:107]
        vels = state[107:125] + state[143:161] + state[179:197]
        if self._n_clips > 1:
            return np.concatenate([np.array(phase),
                                   np.array(root_y),
                                   np.array(rots),
                                   np.array(vels),
                                   np.array(self._one_hot_current_clip)])
        return np.concatenate([np.array(phase),
                                   np.array(root_y),
                                   np.array(rots),
                                   np.array(vels)])
