from pybullet_utils import pd_controller_stable
from deep_mimic.env import humanoid_pose_interpolator_upper
from deep_mimic.env.humanoid_stable_pd_selector import HumanoidStablePDSelector
import math
import numpy as np

chest = 1
neck = 2
rightHip = 3
rightKnee = 4
rightAnkle = 5
rightShoulder = 6
rightElbow = 7
leftHip = 9
leftKnee = 10
leftAnkle = 11
leftShoulder = 12
leftElbow = 13
jointFrictionForce = 0


class HumanoidStablePDUpper(HumanoidStablePDSelector):

    def __init__(self, pybullet_client, mocap_data, timeStep,
                 useFixedBase=True, arg_parser=None, useComReward=False, current_clip=-1):
        self._pybullet_client = pybullet_client
        self._mocap_data = mocap_data # this is a dictionary
        self._arg_parser = arg_parser
        self._n_clips = self._mocap_data.getNumClips()
        self._current_clip = current_clip
        print("LOADING humanoid!")
        flags = self._pybullet_client.URDF_MAINTAIN_LINK_ORDER + self._pybullet_client.URDF_USE_SELF_COLLISION + self._pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self._sim_model = self._pybullet_client.loadURDF(
            "humanoid/humanoid.urdf", [0, 0.889540259, 0],
            globalScaling=0.25,
            useFixedBase=useFixedBase,
            flags=flags)

        self._end_effectors = [5, 8, 11, 14]  # ankle and wrist, both left and right


        self._kin_model = self._pybullet_client.loadURDF(
            "humanoid/humanoid.urdf", [0, 0.85, 0],
            globalScaling=0.25,
            useFixedBase=True,
            flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER)

        self._pybullet_client.changeDynamics(self._sim_model, -1, lateralFriction=0.9)
        for j in range(self._pybullet_client.getNumJoints(self._sim_model)):
            self._pybullet_client.changeDynamics(self._sim_model, j, lateralFriction=0.9)

        self._pybullet_client.changeDynamics(self._sim_model, -1, linearDamping=0, angularDamping=0)

        self._pybullet_client.changeDynamics(self._kin_model, -1, linearDamping=0, angularDamping=0)

        # todo: add feature to disable simulation for a particular object. Until then, disable all collisions
        self._pybullet_client.setCollisionFilterGroupMask(self._kin_model,
                                                          -1,
                                                          collisionFilterGroup=0,
                                                          collisionFilterMask=0)
        self._pybullet_client.changeDynamics(
            self._kin_model,
            -1,
            activationState=self._pybullet_client.ACTIVATION_STATE_SLEEP +
                            self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                            self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
        alpha = 0.4
        self._pybullet_client.changeVisualShape(self._kin_model, -1, rgbaColor=[1, 1, 1, alpha])
        for j in range(self._pybullet_client.getNumJoints(self._kin_model)):
            self._pybullet_client.setCollisionFilterGroupMask(self._kin_model,
                                                              j,
                                                              collisionFilterGroup=0,
                                                              collisionFilterMask=0)
            self._pybullet_client.changeDynamics(
                self._kin_model,
                j,
                activationState=self._pybullet_client.ACTIVATION_STATE_SLEEP +
                                self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                                self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
            self._pybullet_client.changeVisualShape(self._kin_model, j, rgbaColor=[1, 1, 1, alpha])

        self._poseInterpolator = humanoid_pose_interpolator_upper.HumanoidPoseInterpolatorUpper()

        self._stablePD = pd_controller_stable.PDControllerStableMultiDof(self._pybullet_client)
        self._timeStep = timeStep
        self._kpOrg = [
            0, 0, 0,
            0, 0, 0, 0,
            1000, 1000, 1000, 1000,
            100, 100, 100, 100,
            500,
            500, 500, 500, 500,
            400, 400, 400, 400,
            400, 400, 400, 400,
            300,
            500, 500, 500, 500,
            500,
            400, 400, 400, 400,
            400, 400, 400, 400,
            300
        ]
        self._kdOrg = [
            0, 0, 0,
            0, 0, 0, 0,
            100, 100, 100, 100,
            10, 10, 10, 10,
            50,
            50, 50, 50, 50,
            40, 40, 40, 40,
            40, 40, 40, 40,
            30,
            50, 50, 50, 50,
            50,
            40, 40, 40, 40,
            40, 40, 40, 40,
            30
        ]

        self._jointIndicesAll = [
            chest, neck,
            rightHip, rightKnee, rightAnkle,
            rightShoulder, rightElbow,
            leftHip, leftKnee, leftAnkle,
            leftShoulder, leftElbow
        ]
        for j in self._jointIndicesAll:
            self._pybullet_client.setJointMotorControl2(self._sim_model,
                                                        j,
                                                        self._pybullet_client.POSITION_CONTROL,
                                                        targetPosition=0,
                                                        positionGain=0,
                                                        targetVelocity=0,
                                                        force=jointFrictionForce)
            self._pybullet_client.setJointMotorControlMultiDof(
                self._sim_model,
                j,
                self._pybullet_client.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[jointFrictionForce, jointFrictionForce, jointFrictionForce])


            self._pybullet_client.setJointMotorControl2(self._kin_model,
                                                        j,
                                                        self._pybullet_client.POSITION_CONTROL,
                                                        targetPosition=0,
                                                        positionGain=0,
                                                        targetVelocity=0,
                                                        force=0)
            self._pybullet_client.setJointMotorControlMultiDof(
                self._kin_model,
                j,
                self._pybullet_client.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[jointFrictionForce, jointFrictionForce, 0])

        self._jointDofCounts = [
            4, 4,
            4, 1, 4,
            4, 1,
            4, 1, 4,
            4, 1]

        # only those body parts/links are allowed to touch the ground, otherwise the episode terminates
        fall_contact_bodies = []
        if self._arg_parser is not None:
            fall_contact_bodies = self._arg_parser.parse_ints("fall_contact_bodies")
        self._fall_contact_body_parts = fall_contact_bodies

        # [x,y,z] base position and [x,y,z,w] base orientation
        self._totalDofs = 7
        for dof in self._jointDofCounts:
            self._totalDofs += dof
        self.setSimTime(0)

        self._useComReward = useComReward

        self.resetPose()