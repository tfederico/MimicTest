import numpy as np
import math
from deep_mimic.env.env import Env
from deep_mimic.env.action_space import ActionSpace
from pybullet_utils import bullet_client
import time
from deep_mimic.env import motion_capture_data
from deep_mimic.env import humanoid_stable_pd_upper_whole as stable_pd
import data as pybullet_data
import pybullet as p1
import random

from enum import Enum

class InitializationStrategy(Enum):
    """Set how the environment is initialized."""
    START = 0
    RANDOM = 1  # random state initialization (RSI)


class PyBulletDeepMimicEnv(Env):

    def __init__(self, arg_parser=None, enable_draw=False, pybullet_client=None, time_step=1./240,
                 init_strategy=InitializationStrategy.RANDOM, use_com_reward=False):
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
            self._pybullet_client.setGravity(0, -9.8, 0)

            self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
            self._pybullet_client.changeDynamics(self._planeId, linkIndex=-1, lateralFriction=0.9)

            self._mocapData = motion_capture_data.MotionCaptureData()

            motion_file = self._arg_parser.parse_strings('motion_file')
            print("motion_file=", motion_file[0])
            motionPath = pybullet_data.getDataPath() + "/" + motion_file[0]
            print(motionPath)
            self._mocapData.Load(motionPath)
            timeStep = self.timeStep
            useFixedBase = False
            self._humanoid = stable_pd.HumanoidStablePDWholeUpper(pybullet_client=self._pybullet_client,
                                                                  mocap_data=self._mocapData, timeStep=timeStep,
                                                                  useFixedBase=useFixedBase,
                                                                  arg_parser=self._arg_parser,
                                                                  useComReward=False)
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
        #                     state_size += GetStatePoseSize();#(4+3)*(7+16+16) + 1 = 274
        #                     state_size += GetStateVelSize(); #(4+3-1)*(7+16+16) = 234
        #state_size += GetStatePhaseSize();#1
        return 509

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
        ctrl_size = (18+7)+16+16  #numDof
        root_size = 7
        return ctrl_size - root_size

    def build_goal_norm_groups(self, agent_id):
        return np.array([])

    def build_goal_offset(self, agent_id):
        return np.array([])

    def build_goal_scale(self, agent_id):
        return np.array([])

    def build_action_offset(self, agent_id):
        # out_offset = []
        # dofs = [4, 4, 4, 1] + [1] * 16 + [4, 1] + [1] * 16
        # lows = self.build_action_bound_min(-1)
        # highs = self.build_action_bound_max(-1)
        #
        # for i, dof in enumerate(dofs):
        #     if dof == 4:
        #         out_offset += [0.0, 0.0, 0.0, -0.2]
        #     else:
        #         out_offset.append(-0.5*(highs[i]+lows[i]))
        out_offset = [
            0.0000000000, 0.0000000000, 0.0000000000, -0.200000000,
            0.0000000000, 0.0000000000, 0.0000000000, -0.200000000,
            0.00000000, 0.00000000, 0.00000000, -0.2000000,
            -1.5700000
        ] + [-0.785] * 16 + [
            0.00000000, 0.00000000, 0.00000000, -0.2000000,
            -1.5700000
        ] + [-0.785] * 16
        return np.array(out_offset)

    def build_action_scale(self, agent_id):
        # out_scale = []
        # dofs = [4, 4, 4, 1] + [1] * 16 + [4, 1] + [1] * 16
        # lows = self.build_action_bound_min(-1)
        # highs = self.build_action_bound_max(-1)
        # base_dof = 0
        # for i, dof in enumerate(dofs):
        #     if dof == 4:
        #         out_scale += [0.5/(highs[base_dof+k] - lows[base_dof+k]) for k in range(4)]
        #     else:
        #         out_scale.append(2/(highs[base_dof]-lows[base_dof]))
        #     base_dof += dof
        # # out_scale = [2/(h - l) for l, h in zip(lows, highs)]
        out_scale = [
            0.20833333333333, 1.00000000000000, 1.00000000000000, 1.00000000000000,
            0.25000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
            0.079617834394, 1.000000000000, 1.000000000000, 1.000000000000,
            0.159235668789
        ] + [0.31847133758] * 16 + [
            0.079617834394, 1.000000000000, 1.000000000000, 1.000000000000,
            0.159235668789
        ] + [0.31847133758] * 16
        return np.array(out_scale)

    def build_action_bound_min(self, agent_id):
        #see cCtCtrlUtil::BuildBoundsPDSpherical
        out_scale = [-1] * self.get_action_size(agent_id)
        out_scale = [
            -4.79999999999, -1.00000000000, -1.00000000000, -1.00000000000,
            -4.00000000000, -1.00000000000, -1.00000000000, -1.00000000000,
            -12.56000000, -1.000000000, -1.000000000, -1.000000000,
            -4.710000000
        ] + [-2.355] * 16 + [
            -12.56000000, -1.000000000, -1.000000000, -1.000000000,
            -4.710000000
        ] + [-2.355] * 16

        return out_scale

    def build_action_bound_max(self, agent_id):
        out_scale = [1] * self.get_action_size(agent_id)
        out_scale = [
            4.799999999, 1.000000000, 1.000000000, 1.000000000,
            4.000000000, 1.000000000, 1.000000000, 1.000000000,
            12.560000, 1.0000000, 1.0000000, 1.0000000,
            7.8500000
        ] + [3.925] * 16 + [
            12.560000, 1.0000000, 1.0000000, 1.0000000,
            7.8500000
        ] + [3.925] * 16
        return out_scale

    def set_mode(self, mode):
        self._mode = mode

    def record_state(self, agent_id):
        state = self._humanoid.getState()
        phase = [state[0]]
        root_y = [state[1]]
        rots = state[2:23] + state[44:170] + state[191:317]
        vels = state[317:335] + state[353:461] + state[479:587]

        return np.concatenate([np.array(phase),
                               np.array(root_y),
                               np.array(rots),
                               np.array(vels)])

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
                0, 0, 0, 0,
                200, 200, 200, 200,
                50, 50, 50, 50,
                200, 200, 200, 200,
                150,
                90, 90, 90, 90,
                100, 100, 100, 100,
                60] + [50] * 16 + [
                200, 200, 200, 200,
                150,
                90, 90, 90, 90,
                100, 100, 100, 100,
                60
            ] + [50] * 16


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
        if o in keys:
            return keys[ord(key)] & self._pybullet_client.KEY_WAS_TRIGGERED
        return False


def test_pybullet():
    from pybullet_utils.arg_parser import ArgParser
    import time
    from pytorch3d import transforms as t3d
    import torch

    arg_file = "run_humanoid3d_tuning_motion_whole_args.txt"
    arg_parser = ArgParser()
    path = pybullet_data.getDataPath() + "/args/" + arg_file
    succ = arg_parser.load_file(path)
    timeStep = 1. / 30
    _init_strategy = InitializationStrategy.START

    env = PyBulletDeepMimicEnv(arg_parser=arg_parser, enable_draw=True, pybullet_client=None,
                               time_step=timeStep,
                               init_strategy=_init_strategy,
                               use_com_reward=False)


    steps = range(700)

    actions = []
    dofs = [4, 4, 4, 1, 4, 4, 1] + [1] * 16 + [4, 1, 4, 4, 1] + [1] * 16
    for i in steps:
        action = env._mocapData._motion_data['Frames'][env._humanoid._frameNext][8:]

        angle_axis = []
        base_index = 0
        i = 0
        skip = [2, 3, 4, 23, 24, 25]
        for dof in dofs:
            if i not in skip:
                a = action[base_index:base_index + dof]
                if dof == 4:
                    a = t3d.quaternion_to_axis_angle(torch.unsqueeze(torch.tensor(a), 0)).numpy().tolist()[0]
                    norm = math.sqrt(sum([b * b for b in a]))
                    a = [b / norm for b in a]
                    a = [norm] + a

                angle_axis.append(a)
            base_index += dof
            i += 1

        flat_angle_axis = []
        for a in angle_axis:
            flat_angle_axis += a

        action = flat_angle_axis
        actions.append(action)
        env.set_action(0, action)

        env.update(timeStep)
        time.sleep(1/30)

    actions = np.array(actions)

    for i in range(50):
        print(i, min(actions[:, i]), max(actions[:, i]))


if __name__ == '__main__':
    test_pybullet()