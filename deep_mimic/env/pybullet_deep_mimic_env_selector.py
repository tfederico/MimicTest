import numpy as np
import math
from deep_mimic.env.pybullet_deep_mimic_env_multiclip import PyBulletDeepMimicEnvMultiClip, InitializationStrategy, Env
from pybullet_utils import bullet_client
import time
from deep_mimic.env import motion_capture_data_multiclip
from deep_mimic.env import humanoid_stable_pd_selector
import data as pybullet_data
import pybullet as p1
import random


class PyBulletDeepMimicEnvSelector(PyBulletDeepMimicEnvMultiClip):

    def __init__(self, arg_parser=None, enable_draw=False, pybullet_client=None,
                 time_step=1. / 240,
                 init_strategy=InitializationStrategy.RANDOM,
                 use_com_reward = False):
        #super().__init__(arg_parser, enable_draw, pybullet_client, time_step, init_strategy, use_com_reward)
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
            self._humanoid = humanoid_stable_pd_selector.HumanoidStablePDSelector(self._pybullet_client, self._mocapData,
                                                                                  timeStep, useFixedBase, self._arg_parser,
                                                                                  self._use_com_reward, self._current_clip)
            self._isInitialized = True

            self._pybullet_client.setTimeStep(timeStep)
            self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

        # print("numframes = ", self._humanoid._mocap_data.getNumFrames())

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
        #                     state_size += GetStatePoseSize();#106
        #                     state_size += GetStateVelSize(); #(3+3)*numBodyParts=90
        # state_size += GetStatePhaseSize();#1
        # 197
        if self._n_clips > 1:
            return 197 + self._n_clips
        return 197



    def record_state(self, agent_id):
        state = self._humanoid.getState()
        if self._n_clips > 1:
            return np.concatenate([np.array(state), np.array(self._one_hot_current_clip)])
        return np.array(state)


    def calc_reward(self, agent_id):
        kinPose = self._humanoid.computePose(self._humanoid._frameFraction, self._current_clip)
        reward = self._humanoid.getReward(kinPose)

        return reward

    def set_action(self, agent_id, action):
        self.desiredPose = self._humanoid.convertActionToPose(action)
        # we need the target root positon and orientation to be zero, to be compatible with deep mimic
        self.desiredPose[0] = 0
        self.desiredPose[1] = 0
        self.desiredPose[2] = 0
        self.desiredPose[3] = 0
        self.desiredPose[4] = 0
        self.desiredPose[5] = 0
        self.desiredPose[6] = 0



    def update(self, timeStep):
        # print("pybullet_deep_mimic_env:update timeStep=",timeStep," t=",self.t)
        self._pybullet_client.setTimeStep(timeStep)
        self._humanoid._timeStep = timeStep
        self.timeStep = timeStep

        self.t += timeStep
        self.remaining_t -= timeStep
        self._humanoid.setSimTime(self.t)

        if self.desiredPose:

            kinPose = self._humanoid.computePose(self._humanoid._frameFraction, self._current_clip)
            self._humanoid.initializePose(self._humanoid._poseInterpolator,
                                          self._humanoid._kin_model,
                                          initBase=True)
            # print("desiredPositions=",self.desiredPose[i])
            maxForces = [
                0, 0, 0,
                0, 0, 0, 0,
                200, 200, 200, 200,
                50, 50, 50, 50,
                200, 200, 200, 200,
                150,
                90, 90, 90, 90,
                100, 100, 100, 100,
                60,
                200, 200, 200, 200,
                150,
                90, 90, 90, 90,
                100, 100, 100, 100,
                60
            ]

            self._humanoid.computeAndApplyPDForces(self.desiredPose, maxForces=maxForces)

            self._pybullet_client.stepSimulation()



    def is_episode_end(self):
        isEnded = self._humanoid.terminates()
        # also check maximum time, 20 seconds (todo get from file)
        # print("self.t=",self.t)
        if (self.remaining_t <= 0):
            isEnded = True
        return isEnded

    def get_num_clips(self):
        return self._n_clips

    def get_current_clip_num(self):
        return self._current_clip

    def change_current_clip(self):
        assert self._n_clips > 1, "To change clip, you need to have more than 1 clip"
        #print("Old clip: ", self._current_clip)
        old_clip = self._current_clip
        while(old_clip == self._current_clip):
            self._current_clip = random.randint(0, self._n_clips - 1)
        #print("New clip: ", self._current_clip)
        self._one_hot_current_clip = np.eye(self._n_clips)[self._current_clip]
        self.t = 0
        self._humanoid.setSimTime(self.t)
        self._humanoid.change_current_clip(self._current_clip)
