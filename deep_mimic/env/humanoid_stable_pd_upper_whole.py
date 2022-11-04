import numpy
import torch
from pybullet_utils import pd_controller_stable
from deep_mimic.env import humanoid_pose_interpolator_upper_whole
import math
import numpy as np

chest = 1
neck = 2
rightHip = 3
rightKnee = 4
rightAnkle = 5
rightShoulder = 6
rightElbow = 7
rightWrist = 8
rightThumbProximal = 9
rightThumbIntermediate = 10
rightThumbDistal = 11
rightIndexProximal = 12
rightIndexIntermediate = 13
rightIndexDistal = 14
rightMiddleProximal = 15
rightMiddleIntermediate = 16
rightMiddleDistal = 17
rightRingProximal = 18
rightRingIntermediate = 19
rightRingDistal = 20
rightPinkyProximal = 21
rightPinkyIntermediate = 22
rightPinkyDistal = 23
leftHip = 24
leftKnee = 25
leftAnkle = 26
leftShoulder = 27
leftElbow = 28
leftWrist = 29
leftThumbProximal = 30
leftThumbIntermediate = 31
leftThumbDistal = 32
leftIndexProximal = 33
leftIndexIntermediate = 34
leftIndexDistal = 35
leftMiddleProximal = 36
leftMiddleIntermediate = 37
leftMiddleDistal = 38
leftRingProximal = 39
leftRingIntermediate = 40
leftRingDistal = 41
leftPinkyProximal = 42
leftPinkyIntermediate = 43
leftPinkyDistal = 44

jointFrictionForce = 0


class HumanoidStablePDWholeUpper(object):

    def __init__(self, pybullet_client, mocap_data, timeStep,
                 useFixedBase=True, arg_parser=None, useComReward=False):

        self._pybullet_client = pybullet_client
        self._mocap_data = mocap_data
        self._arg_parser = arg_parser
        print("LOADING humanoid!")
        flags = self._pybullet_client.URDF_MAINTAIN_LINK_ORDER + self._pybullet_client.URDF_USE_SELF_COLLISION + self._pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self._sim_model = self._pybullet_client.loadURDF(
            "humanoid/whole_upper_humanoid.urdf", [0, 0.889540259, 0],
            globalScaling=0.25,
            useFixedBase=useFixedBase,
            flags=flags)

        # self._pybullet_client.setCollisionFilterGroupMask(self._sim_model,-1,collisionFilterGroup=0,collisionFilterMask=0)
        # for j in range (self._pybullet_client.getNumJoints(self._sim_model)):
        #  self._pybullet_client.setCollisionFilterGroupMask(self._sim_model,j,collisionFilterGroup=0,collisionFilterMask=0)

        self._end_effectors = [5, 8, 26, 30]  # ankle and wrist, both left and right

        self._kin_model = self._pybullet_client.loadURDF(
            "humanoid/whole_upper_humanoid.urdf", [0, 0.85, 0],
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

        self._poseInterpolator = humanoid_pose_interpolator_upper_whole.WholeHumanoidPoseInterpolator()

        # for i in range(self._mocap_data.getNumFrames() - 1):
        #     frameData = self._mocap_data._motion_data['Frames'][i]

        self._stablePD = pd_controller_stable.PDControllerStableMultiDof(self._pybullet_client)
        self._timeStep = timeStep
        self._kpOrg = [
            0, 0, 0,
            0, 0, 0, 0,
            1000, 1000, 1000, 1000,
            100, 100, 100, 100,
            500, 500, 500, 500,
            500,
            400, 400, 400, 400,
            400, 400, 400, 400,
            300] + [50] * 16 + [500, 500, 500, 500,
            500,
            400, 400, 400, 400,
            400, 400, 400, 400,
            300
        ] + [50] * 16
        self._kdOrg = [
            0, 0, 0,
            0, 0, 0, 0,
            100, 100, 100, 100,
            10, 10, 10, 10,
            50, 50, 50, 50,
            50,
            8, 8, 8, 8,
            8, 8, 8, 8,
            6] + [0.5] * 16 + [50, 50, 50, 50,
            50,
            8, 8, 8, 8,
            8, 8, 8, 8,
            6
        ] + [0.5] * 16

        self._jointIndicesAll = [
            chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow,
            rightWrist,
            rightThumbProximal, rightThumbIntermediate, rightThumbDistal,
            rightIndexProximal, rightIndexIntermediate, rightIndexDistal,
            rightMiddleProximal, rightMiddleIntermediate, rightMiddleDistal,
            rightRingProximal, rightRingIntermediate, rightRingDistal,
            rightPinkyProximal, rightPinkyIntermediate, rightPinkyDistal,
            leftHip, leftKnee, leftAnkle, leftShoulder, leftElbow,
            leftWrist,
            leftThumbProximal, leftThumbIntermediate, leftThumbDistal,
            leftIndexProximal, leftIndexIntermediate, leftIndexDistal,
            leftMiddleProximal, leftMiddleIntermediate, leftMiddleDistal,
            leftRingProximal, leftRingIntermediate, leftRingDistal,
            leftPinkyProximal, leftPinkyIntermediate, leftPinkyDistal
        ]
        for j in self._jointIndicesAll:
            # self._pybullet_client.setJointMotorControlMultiDof(self._sim_model, j, self._pybullet_client.POSITION_CONTROL, force=[1,1,1])
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

        self._jointDofCounts = [4, 4, 4, 1, 4, 4, 1] + [1] * 16 + [4, 1, 4, 4, 1] + [1] * 16

        # only those body parts/links are allowed to touch the ground, otherwise the episode terminates
        fall_contact_bodies = []
        if self._arg_parser is not None:
            fall_contact_bodies = self._arg_parser.parse_ints("fall_contact_bodies")
        self._fall_contact_body_parts = fall_contact_bodies

        # [x,y,z] base position and [x,y,z,w] base orientation!
        self._totalDofs = 7
        for dof in self._jointDofCounts:
            self._totalDofs += dof
        self.setSimTime(0)

        self._useComReward = useComReward

        self.resetPose()

    def resetPose(self):
        # print("resetPose with self._frame=", self._frame, " and self._frameFraction=",self._frameFraction)
        pose = self.computePose(self._frameFraction)
        self.initializePose(self._poseInterpolator, self._sim_model, initBase=True)
        self.initializePose(self._poseInterpolator, self._kin_model, initBase=False)

    def initializePose(self, pose, phys_model, initBase, initializeVelocity=True):

        if initializeVelocity:
            if initBase:
                self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose._basePos,
                                                                      pose._baseOrn)
                self._pybullet_client.resetBaseVelocity(phys_model, pose._baseLinVel, pose._baseAngVel)

            indices = [
                chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow,
                rightWrist,
                rightThumbProximal, rightThumbIntermediate, rightThumbDistal,
                rightIndexProximal, rightIndexIntermediate, rightIndexDistal,
                rightMiddleProximal, rightMiddleIntermediate, rightMiddleDistal,
                rightRingProximal, rightRingIntermediate, rightRingDistal,
                rightPinkyProximal, rightPinkyIntermediate, rightPinkyDistal,
                leftHip, leftKnee, leftAnkle, leftShoulder, leftElbow,
                leftWrist,
                leftThumbProximal, leftThumbIntermediate, leftThumbDistal,
                leftIndexProximal, leftIndexIntermediate, leftIndexDistal,
                leftMiddleProximal, leftMiddleIntermediate, leftMiddleDistal,
                leftRingProximal, leftRingIntermediate, leftRingDistal,
                leftPinkyProximal, leftPinkyIntermediate, leftPinkyDistal
            ]
            jointPositions = [
                pose._chestRot, pose._neckRot, pose._rightHipRot, pose._rightKneeRot,
                pose._rightAnkleRot, pose._rightShoulderRot, pose._rightElbowRot, pose._rightWristRot,
                pose._rightThumbProx, pose._rightThumbInter, pose._rightThumbDist,
                pose._rightIndexProx, pose._rightIndexInter, pose._rightIndexDist,
                pose._rightMiddleProx, pose._rightMiddleInter, pose._rightMiddleDist,
                pose._rightRingProx, pose._rightRingInter, pose._rightRingDist,
                pose._rightPinkieProx, pose._rightPinkieInter, pose._rightPinkieDist,
                pose._leftHipRot,
                pose._leftKneeRot, pose._leftAnkleRot, pose._leftShoulderRot, pose._leftElbowRot, pose._leftWristRot,
                pose._leftThumbProx, pose._leftThumbInter, pose._leftThumbDist,
                pose._leftIndexProx, pose._leftIndexInter, pose._leftIndexDist,
                pose._leftMiddleProx, pose._leftMiddleInter, pose._leftMiddleDist,
                pose._leftRingProx, pose._leftRingInter, pose._leftRingDist,
                pose._leftPinkieProx, pose._leftPinkieInter, pose._leftPinkieDist
            ]

            jointVelocities = [
                pose._chestVel, pose._neckVel, pose._rightHipVel, pose._rightKneeVel,
                pose._rightAnkleVel, pose._rightShoulderVel, pose._rightElbowVel, pose._rightWristVel,
                pose._rightThumbProxVel, pose._rightThumbInterVel, pose._rightThumbDistVel,
                pose._rightIndexProxVel, pose._rightIndexInterVel, pose._rightIndexDistVel,
                pose._rightMiddleProxVel, pose._rightMiddleInterVel, pose._rightMiddleDistVel,
                pose._rightRingProxVel, pose._rightRingInterVel, pose._rightRingDistVel,
                pose._rightPinkieProxVel, pose._rightPinkieInterVel, pose._rightPinkieDistVel,
                pose._leftHipVel, pose._leftKneeVel, pose._leftAnkleVel, pose._leftShoulderVel,
                pose._leftElbowVel, pose._leftWristVel,
                pose._leftThumbProxVel, pose._leftThumbInterVel, pose._leftThumbDistVel,
                pose._leftIndexProxVel, pose._leftIndexInterVel, pose._leftIndexDistVel,
                pose._leftMiddleProxVel, pose._leftMiddleInterVel, pose._leftMiddleDistVel,
                pose._leftRingProxVel, pose._leftRingInterVel, pose._leftRingDistVel,
                pose._leftPinkieProxVel, pose._leftPinkieInterVel, pose._leftPinkieDistVel
            ]
            self._pybullet_client.resetJointStatesMultiDof(phys_model, indices,
                                                           jointPositions, jointVelocities)
        else:
            if initBase:
                self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose._basePos,
                                                                      pose._baseOrn)

            indices = [
                chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow,
                rightWrist,
                rightThumbProximal, rightThumbIntermediate, rightThumbDistal,
                rightIndexProximal, rightIndexIntermediate, rightIndexDistal,
                rightMiddleProximal, rightMiddleIntermediate, rightMiddleDistal,
                rightRingProximal, rightRingIntermediate, rightRingDistal,
                rightPinkyProximal, rightPinkyIntermediate, rightPinkyDistal,
                leftHip, leftKnee, leftAnkle, leftShoulder, leftElbow,
                leftWrist,
                leftThumbProximal, leftThumbIntermediate, leftThumbDistal,
                leftIndexProximal, leftIndexIntermediate, leftIndexDistal,
                leftMiddleProximal, leftMiddleIntermediate, leftMiddleDistal,
                leftRingProximal, leftRingIntermediate, leftRingDistal,
                leftPinkyProximal, leftPinkyIntermediate, leftPinkyDistal
            ]
            jointPositions = [
                pose._chestRot, pose._neckRot, pose._rightHipRot, pose._rightKneeRot,
                pose._rightAnkleRot, pose._rightShoulderRot, pose._rightElbowRot, pose._rightWristRot,
                pose._rightThumbProx, pose._rightThumbInter, pose._rightThumbDist,
                pose._rightIndexProx, pose._rightIndexInter, pose._rightIndexDist,
                pose._rightMiddleProx, pose._rightMiddleInter, pose._rightMiddleDist,
                pose._rightRingProx, pose._rightRingInter, pose._rightRingDist,
                pose._rightPinkieProx, pose._rightPinkieInter, pose._rightPinkieDist,
                pose._leftHipRot,
                pose._leftKneeRot, pose._leftAnkleRot, pose._leftShoulderRot, pose._leftElbowRot, pose._leftWristRot,
                pose._leftThumbProx, pose._leftThumbInter, pose._leftThumbDist,
                pose._leftIndexProx, pose._leftIndexInter, pose._leftIndexDist,
                pose._leftMiddleProx, pose._leftMiddleInter, pose._leftMiddleDist,
                pose._leftRingProx, pose._leftRingInter, pose._leftRingDist,
                pose._leftPinkieProx, pose._leftPinkieInter, pose._leftPinkieDist
            ]
            self._pybullet_client.resetJointStatesMultiDof(phys_model, indices, jointPositions)

    def calcCycleCount(self, simTime, cycleTime):
        phases = simTime / cycleTime
        count = math.floor(phases)

        return count

    def getCycleTime(self):
        keyFrameDuration = self._mocap_data.getKeyFrameDuration()
        cycleTime = keyFrameDuration * (self._mocap_data.getNumFrames() - 1)
        return cycleTime

    def setSimTime(self, t):
        self._simTime = t
        keyFrameDuration = self._mocap_data.getKeyFrameDuration()
        cycleTime = self.getCycleTime()
        self._cycleCount = self.calcCycleCount(t, cycleTime)
        frameTime = t - self._cycleCount * cycleTime
        if (frameTime < 0):
            frameTime += cycleTime
        self._frame = int(frameTime / keyFrameDuration)
        self._frameNext = self._frame + 1
        if (self._frameNext >= self._mocap_data.getNumFrames()):
            self._frameNext = self._frame
        self._frameFraction = (frameTime - self._frame * keyFrameDuration) / (keyFrameDuration)

    def computeCycleOffset(self):
        lastFrame = self._mocap_data.getNumFrames() - 1
        frameData = self._mocap_data._motion_data['Frames'][0]
        frameDataNext = self._mocap_data._motion_data['Frames'][lastFrame]

        basePosStart = [frameData[1], frameData[2], frameData[3]]
        basePosEnd = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
        self._cycleOffset = [
            basePosEnd[0] - basePosStart[0], basePosEnd[1] - basePosStart[1],
            basePosEnd[2] - basePosStart[2]
        ]
        return self._cycleOffset

    def computePose(self, frameFraction):
        frameData = self._mocap_data._motion_data['Frames'][self._frame]
        frameDataNext = self._mocap_data._motion_data['Frames'][self._frameNext]

        self._poseInterpolator.Slerp(frameFraction, frameData, frameDataNext, self._pybullet_client)
        self.computeCycleOffset()
        oldPos = self._poseInterpolator._basePos
        self._poseInterpolator._basePos = [
            oldPos[0] + self._cycleCount * self._cycleOffset[0],
            oldPos[1] + self._cycleCount * self._cycleOffset[1],
            oldPos[2] + self._cycleCount * self._cycleOffset[2]
        ]
        pose = self._poseInterpolator.GetPose()

        return pose

    def convertActionToPose(self, action):
        pose = self._poseInterpolator.ConvertFromAction(self._pybullet_client, action)
        return pose

    def computeAndApplyPDForces(self, desiredPositions, maxForces):
        dofIndex = 7
        scaling = 1
        indices = []
        forces = []
        targetPositions = []
        targetVelocities = []
        kps = []
        kds = []

        for index in range(len(self._jointIndicesAll)):
            jointIndex = self._jointIndicesAll[index]
            indices.append(jointIndex)
            kps.append(self._kpOrg[dofIndex])
            kds.append(self._kdOrg[dofIndex])

            if self._jointDofCounts[index] == 4:
                force = [
                    scaling * maxForces[dofIndex + 0],
                    scaling * maxForces[dofIndex + 1],
                    scaling * maxForces[dofIndex + 2]
                ]
                targetVelocity = [0, 0, 0]
                targetPosition = [
                    desiredPositions[dofIndex + 0],
                    desiredPositions[dofIndex + 1],
                    desiredPositions[dofIndex + 2],
                    desiredPositions[dofIndex + 3]
                ]
            if self._jointDofCounts[index] == 1:
                force = [scaling * maxForces[dofIndex]]
                targetPosition = [desiredPositions[dofIndex + 0]]
                targetVelocity = [0]
            forces.append(force)
            targetPositions.append(targetPosition)
            targetVelocities.append(targetVelocity)
            dofIndex += self._jointDofCounts[index]

        self._pybullet_client.setJointMotorControlMultiDofArray(self._sim_model,
                                                                indices,
                                                                self._pybullet_client.STABLE_PD_CONTROL,
                                                                targetPositions=targetPositions,
                                                                targetVelocities=targetVelocities,
                                                                forces=forces,
                                                                positionGains=kps,
                                                                velocityGains=kds,
                                                                )


    def getPhase(self):
        keyFrameDuration = self._mocap_data.getKeyFrameDuration()
        cycleTime = keyFrameDuration * (self._mocap_data.getNumFrames() - 1)
        phase = self._simTime / cycleTime
        phase = math.fmod(phase, 1.0)
        if (phase < 0):
            phase += 1
        return phase

    def buildHeadingTrans(self, rootOrn):
        # align root transform 'forward' with world-space x axis
        refDir = [1, 0, 0]
        rotVec = self._pybullet_client.rotateVector(rootOrn, refDir)
        heading = math.atan2(-rotVec[2], rotVec[0])
        headingOrn = self._pybullet_client.getQuaternionFromAxisAngle([0, 1, 0], -heading)
        return headingOrn

    def buildOriginTrans(self):
        rootPos, rootOrn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
        invRootPos = [-rootPos[0], 0, -rootPos[2]]
        headingOrn = self.buildHeadingTrans(rootOrn)
        invOrigTransPos, invOrigTransOrn = self._pybullet_client.multiplyTransforms([0, 0, 0],
                                                                                    headingOrn,
                                                                                    invRootPos,
                                                                                    [0, 0, 0, 1])

        return invOrigTransPos, invOrigTransOrn

    def getState(self):

        stateVector = []
        phase = self.getPhase()
        stateVector.append(phase)

        rootTransPos, rootTransOrn = self.buildOriginTrans()
        basePos, baseOrn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
        rootPosRel, dummy = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                     basePos, [0, 0, 0, 1])
        localPos, localOrn = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                      basePos, baseOrn)


        stateVector.append(rootPosRel[1])

        self.pb2dmJoints = range(45)

        linkIndicesSim = []
        for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
            linkIndicesSim.append(self.pb2dmJoints[pbJoint])

        linkStatesSim = self._pybullet_client.getLinkStates(self._sim_model, linkIndicesSim,
                                                            computeForwardKinematics=True, computeLinkVelocity=True)

        for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
            j = self.pb2dmJoints[pbJoint]
            ls = linkStatesSim[pbJoint]
            linkPos = ls[0]
            linkOrn = ls[1]
            linkPosLocal, linkOrnLocal = self._pybullet_client.multiplyTransforms(
                rootTransPos, rootTransOrn, linkPos, linkOrn)
            if (linkOrnLocal[3] < 0):
                linkOrnLocal = [-linkOrnLocal[0], -linkOrnLocal[1], -linkOrnLocal[2], -linkOrnLocal[3]]
            linkPosLocal = [
                linkPosLocal[0] - rootPosRel[0], linkPosLocal[1] - rootPosRel[1],
                linkPosLocal[2] - rootPosRel[2]
            ]
            for l in linkPosLocal:
                stateVector.append(l)
            # re-order the quaternion, DeepMimic uses w,x,y,z

            if (linkOrnLocal[3] < 0):
                linkOrnLocal[0] *= -1
                linkOrnLocal[1] *= -1
                linkOrnLocal[2] *= -1
                linkOrnLocal[3] *= -1

            stateVector.append(linkOrnLocal[3])
            stateVector.append(linkOrnLocal[0])
            stateVector.append(linkOrnLocal[1])
            stateVector.append(linkOrnLocal[2])

        for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
            j = self.pb2dmJoints[pbJoint]
            ls = linkStatesSim[pbJoint]
            linkLinVel = ls[6]
            linkAngVel = ls[7]
            linkLinVelLocal, unused = self._pybullet_client.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                               linkLinVel, [0, 0, 0, 1])
            linkAngVelLocal, unused = self._pybullet_client.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                               linkAngVel, [0, 0, 0, 1])

            for l in linkLinVelLocal:
                stateVector.append(l)
            for l in linkAngVelLocal:
                stateVector.append(l)

        return stateVector

    def terminates(self):
        # check if any non-allowed body part hits the ground
        terminates = False
        pts = self._pybullet_client.getContactPoints()
        for p in pts:
            part = -1
            # ignore self-collision
            if (p[1] == p[2]):
                continue
            if (p[1] == self._sim_model):
                part = p[3]
            if (p[2] == self._sim_model):
                part = p[4]
            if (part >= 0 and part in self._fall_contact_body_parts):
                # print("terminating part:", part)
                terminates = True

        return terminates

    def quatMul(self, q1, q2):
        return [
            q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
            q1[3] * q2[1] + q1[1] * q2[3] + q1[2] * q2[0] - q1[0] * q2[2],
            q1[3] * q2[2] + q1[2] * q2[3] + q1[0] * q2[1] - q1[1] * q2[0],
            q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
        ]

    def calcRootAngVelErr(self, vel0, vel1):
        diff = [vel0[0] - vel1[0], vel0[1] - vel1[1], vel0[2] - vel1[2]]
        return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

    def calcRootRotDiff(self, orn0, orn1):
        orn0Conj = [-orn0[0], -orn0[1], -orn0[2], orn0[3]]
        q_diff = self.quatMul(orn1, orn0Conj)
        axis, angle = self._pybullet_client.getAxisAngleFromQuaternion(q_diff)
        return angle * angle

    def calcSubErrors(self, simJointStates, kinJointStates, linkStatesSim, linkStatesKin, jointIndices, mJointWeights):

        pose_err = 0.0
        vel_err = 0.0
        end_eff_err = 0.0
        num_end_effs = 0

        num_joints = len(jointIndices)

        for j in range(num_joints):
            curr_pose_err = 0
            curr_vel_err = 0
            w = mJointWeights[j]
            simJointInfo = simJointStates[j]

            kinJointInfo = kinJointStates[j]

            if (len(simJointInfo[0]) == 1):
                angle = simJointInfo[0][0] - kinJointInfo[0][0]
                curr_pose_err = angle * angle
                velDiff = simJointInfo[1][0] - kinJointInfo[1][0]
                curr_vel_err = velDiff * velDiff
            if (len(simJointInfo[0]) == 4):
                # print("quaternion diff")
                diffQuat = self._pybullet_client.getDifferenceQuaternion(simJointInfo[0], kinJointInfo[0])
                axis, angle = self._pybullet_client.getAxisAngleFromQuaternion(diffQuat)
                curr_pose_err = angle * angle
                diffVel = [
                    simJointInfo[1][0] - kinJointInfo[1][0], simJointInfo[1][1] - kinJointInfo[1][1],
                    simJointInfo[1][2] - kinJointInfo[1][2]
                ]
                curr_vel_err = diffVel[0] * diffVel[0] + diffVel[1] * diffVel[1] + diffVel[2] * diffVel[2]

            pose_err += w * curr_pose_err
            vel_err += w * curr_vel_err

            is_end_eff = jointIndices[j] in self._end_effectors

            if is_end_eff:
                linkStateSim = linkStatesSim[j]
                linkStateKin = linkStatesKin[j]

                linkPosSim = linkStateSim[0]
                linkPosKin = linkStateKin[0]
                linkPosDiff = [
                    linkPosSim[0] - linkPosKin[0], linkPosSim[1] - linkPosKin[1],
                    linkPosSim[2] - linkPosKin[2]
                ]
                curr_end_err = linkPosDiff[0] * linkPosDiff[0] + linkPosDiff[1] * linkPosDiff[
                    1] + linkPosDiff[2] * linkPosDiff[2]
                end_eff_err += curr_end_err
                num_end_effs += 1

        return pose_err, vel_err, end_eff_err, num_end_effs

    def getReward(self, pose):
        """Compute and return the pose-based reward."""
        # from DeepMimic double cSceneImitate::CalcRewardImitate
        # todo: compensate for ground height in some parts, once we move to non-flat terrain
        # not values from the paper, but from the published code.
        body_w = 0.3
        hands_w = 0.2
        body_vel_w = 0.03
        hands_vel_w = 0.02
        end_eff_w = 0.15
        # does not exist in paper
        root_w = 0.2
        if self._useComReward:
            com_w = 0.1
        else:
            com_w = 0

        total_w = body_w + hands_w + body_vel_w + hands_vel_w + end_eff_w + root_w + com_w
        body_w /= total_w
        hands_w /= total_w
        body_vel_w /= total_w
        hands_vel_w /= total_w
        end_eff_w /= total_w
        root_w /= total_w
        com_w /= total_w

        body_scale = 2
        hands_scale = 0.2
        body_vel_scale = 0.005
        hands_vel_scale = 0.0001
        end_eff_scale = 40
        root_scale = 5
        com_scale = 10
        err_scale = 1  # error scale

        reward = 0

        body_err = 0
        hands_err = 0
        body_vel_err = 0
        hands_vel_err = 0
        end_eff_err = 0
        root_err = 0
        com_err = 0
        heading_err = 0

        if self._useComReward:
            comSim, comSimVel = self.computeCOMposVel(self._sim_model)
            comKin, comKinVel = self.computeCOMposVel(self._kin_model)

        root_id = 0

        num_end_effs = len(self._end_effectors)
        num_joints = 45

        mJointWeights = [1] * num_joints

        root_rot_w = mJointWeights[root_id]
        rootPosSim, rootOrnSim = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
        rootPosKin, rootOrnKin = self._pybullet_client.getBasePositionAndOrientation(self._kin_model)
        linVelSim, angVelSim = self._pybullet_client.getBaseVelocity(self._sim_model)
        # don't read the velocities from the kinematic model (they are zero), use the pose interpolator velocity
        # see also issue https://github.com/bulletphysics/bullet3/issues/2401
        linVelKin = self._poseInterpolator._baseLinVel
        angVelKin = self._poseInterpolator._baseAngVel

        root_rot_err = self.calcRootRotDiff(rootOrnSim, rootOrnKin)
        body_err += root_rot_w * root_rot_err

        root_vel_diff = [
            linVelSim[0] - linVelKin[0], linVelSim[1] - linVelKin[1], linVelSim[2] - linVelKin[2]
        ]
        root_vel_err = root_vel_diff[0] * root_vel_diff[0] + root_vel_diff[1] * root_vel_diff[
            1] + root_vel_diff[2] * root_vel_diff[2]

        root_ang_vel_err = self.calcRootAngVelErr(angVelSim, angVelKin)
        body_vel_err += root_rot_w * root_ang_vel_err

        jointIndices = range(num_joints)
        bodyJointIndices = list(range(rightWrist)) + list(range(leftHip, leftWrist))
        handsJointIndices = list(range(rightWrist, rightPinkyDistal+1)) + list(range(leftWrist, leftPinkyDistal+1))

        simJointStates = self._pybullet_client.getJointStatesMultiDof(self._sim_model, jointIndices)
        kinJointStates = self._pybullet_client.getJointStatesMultiDof(self._kin_model, jointIndices)

        linkStatesSim = self._pybullet_client.getLinkStates(self._sim_model, jointIndices)
        linkStatesKin = self._pybullet_client.getLinkStates(self._kin_model, jointIndices)



        body_results = self.calcSubErrors([simJointStates[b] for b in bodyJointIndices],
                                          [kinJointStates[b] for b in bodyJointIndices],
                                          [linkStatesSim[b] for b in bodyJointIndices],
                                          [linkStatesKin[b] for b in bodyJointIndices],
                                          bodyJointIndices, [mJointWeights[b] for b in bodyJointIndices])
        body_pose_err, b_vel_err, body_end_eff_err, body_num_end_effs = body_results
        body_err += body_pose_err
        body_vel_err += b_vel_err
        hands_results = self.calcSubErrors([simJointStates[h] for h in handsJointIndices],
                                           [kinJointStates[h] for h in handsJointIndices],
                                           [linkStatesSim[h] for h in handsJointIndices],
                                           [linkStatesKin[h] for h in handsJointIndices],
                                           handsJointIndices, [mJointWeights[h] for h in handsJointIndices])
        hands_err, hands_vel_err, hands_end_eff_err, hands_num_end_effs = hands_results

        if (num_end_effs > 0):
            end_eff_err = (body_end_eff_err+hands_end_eff_err)/(body_num_end_effs+hands_num_end_effs)

        root_pos_diff = [
            rootPosSim[0] - rootPosKin[0], rootPosSim[1] - rootPosKin[1], rootPosSim[2] - rootPosKin[2]
        ]
        root_pos_err = root_pos_diff[0] * root_pos_diff[0] + root_pos_diff[1] * root_pos_diff[
            1] + root_pos_diff[2] * root_pos_diff[2]

        root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_ang_vel_err


        if self._useComReward:
            com_err = 0.1 * np.sum(np.square(comKinVel - comSimVel))

        body_reward = math.exp(-err_scale * body_scale * body_err)
        hands_reward = math.exp(-err_scale * hands_scale * hands_err)
        body_vel_reward = math.exp(-err_scale * body_vel_scale * body_vel_err)
        hands_vel_reward = math.exp(-err_scale * hands_vel_scale * hands_vel_err)
        end_eff_reward = math.exp(-err_scale * end_eff_scale * end_eff_err)
        root_reward = math.exp(-err_scale * root_scale * root_err)
        com_reward = math.exp(-err_scale * com_scale * com_err)

        # reward = body_w * body_reward + body_vel_w * body_vel_reward + hands_w * hands_reward \
        #          + hands_vel_w * hands_vel_reward + end_eff_w * end_eff_reward + root_w * root_reward + com_w * com_reward
        reward = body_reward * body_vel_reward * hands_reward * hands_vel_reward * end_eff_reward * root_reward
        if self._useComReward:
            reward *= com_reward

        info_rew = dict(
            body_pose_reward=body_reward,
            body_vel_reward=body_vel_reward,
            hands_pose_reward=hands_reward,
            hands_vel_reward=hands_vel_reward,
            end_eff_reward=end_eff_reward,
            root_reward=root_reward,
            imitation_reward=reward
        )

        info_errs = dict(
            body_pose_err=body_err,
            body_vel_err=body_vel_err,
            hands_pose_err=hands_err,
            hands_vel_err=hands_vel_err,
            end_eff_err=end_eff_err,
            root_err=root_err,
        )

        if self._useComReward:
            info_rew['com_reward'] = com_reward
            info_errs['com_err'] = com_err

        # store reward/err info for safe keeping
        self._info_rew = info_rew
        self._info_err = info_errs

        return reward

    def computeCOMposVel(self, uid: int):
        """Compute center-of-mass position and velocity."""
        pb = self._pybullet_client
        num_joints = 45
        jointIndices = range(num_joints)
        link_states = pb.getLinkStates(uid, jointIndices, computeLinkVelocity=1)
        link_pos = np.array([s[0] for s in link_states])
        link_vel = np.array([s[-2] for s in link_states])
        tot_mass = 0.
        masses = []
        for j in jointIndices:
            mass_, *_ = pb.getDynamicsInfo(uid, j)
            masses.append(mass_)
            tot_mass += mass_
        masses = np.asarray(masses)[:, None]
        com_pos = np.sum(masses * link_pos, axis=0) / tot_mass
        com_vel = np.sum(masses * link_vel, axis=0) / tot_mass
        return com_pos, com_vel


def tune_controller():
    import pybullet as p1
    from pybullet_utils import bullet_client
    import data as pybullet_data
    from pybullet_utils.arg_parser import ArgParser
    from deep_mimic.env.pybullet_deep_mimic_env_hand import InitializationStrategy
    from deep_mimic.env import motion_capture_data


    arg_file = "run_humanoid3d_00433_args.txt"
    arg_parser = ArgParser()
    path = pybullet_data.getDataPath() + "/args/" + arg_file
    succ = arg_parser.load_file(path)
    timeStep = 1. / 240
    _init_strategy = InitializationStrategy.START
    _pybullet_client = bullet_client.BulletClient(connection_mode=p1.GUI)
    # # disable 'GUI' since it slows down a lot on Mac OSX and some other platforms
    _pybullet_client.configureDebugVisualizer(_pybullet_client.COV_ENABLE_GUI, 0)
    _pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    z2y = _pybullet_client.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
    _planeId = _pybullet_client.loadURDF("plane_implicit.urdf", [0, 0, 0],
                                           z2y,
                                           useMaximalCoordinates=True)
    _pybullet_client.configureDebugVisualizer(_pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
    _pybullet_client.setGravity(0, -9.8, 0)
    _pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
    _mocapData = motion_capture_data.MotionCaptureData()
    motion_file = arg_parser.parse_strings('motion_file')
    print(motion_file)
    print("motion_file=", motion_file[0])
    motionPath = pybullet_data.getDataPath() + "/" + motion_file[0]
    print(motionPath)
    _mocapData.Load(motionPath)
    timeStep = timeStep
    useFixedBase = False

    _humanoid = HumanoidStablePDWholeUpper(_pybullet_client, _mocapData, timeStep, useFixedBase, arg_parser)

    _pybullet_client.setTimeStep(timeStep)
    _pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

    startTime = 0

    t = startTime
    _humanoid.setSimTime(startTime)
    _humanoid.resetPose()
    # this clears the contact points. Todo: add API to explicitly clear all contact points?

    # _pybullet_client.stepSimulation()
    _humanoid.resetPose()

    needs_update_time = t - 1  # force update

    import time
    from pytorch3d import transforms as t3d

    errors = []
    body_errors = []
    hands_errors = []
    rewards = []
    body_rewards = []
    hands_rewards = []
    body_vel_errors = []
    hands_vel_errors = []
    body_vel_rewards = []
    hands_vel_rewards = []

    steps = range(798)
    dofs = [4, 4, 4, 1, 4, 4, 1] + [1] * 16 + [4, 1, 4, 4, 1] + [1] * 16
    for i in steps:
        # print(_humanoid._frameNext)
        action = _mocapData._motion_data['Frames'][_humanoid._frameNext][1:]
        action = action[7:]

        angle_axis = []
        base_index = 0
        skip = [2, 3, 4, 23, 24, 25]
        for j, dof in enumerate(dofs):
            if j not in skip:
                a = action[base_index:base_index+dof]
                if dof == 4:
                    # a = a[1:] + [a[0]]
                    a = t3d.quaternion_to_axis_angle(torch.unsqueeze(torch.tensor(a), 0)).numpy().tolist()[0]
                    norm = math.sqrt(sum([b*b for b in a]))
                    a = [b/norm for b in a]
                    a = [norm] + a

                angle_axis.append(a)
            base_index += dof

        flat_angle_axis = []
        for a in angle_axis:
            flat_angle_axis += a

        action = flat_angle_axis
        desired_pose = _humanoid.convertActionToPose(action)

        desired_pose[:7] = [0] * 7


        _pybullet_client.setTimeStep(timeStep)
        _humanoid._timeStep = timeStep
        t += timeStep
        _humanoid.setSimTime(t)
        kinPose = _humanoid.computePose(_humanoid._frameFraction)

        _humanoid.getReward(kinPose)
        errors.append(_humanoid._info_err)
        body_errors.append(_humanoid._info_err["body_pose_err"])
        hands_errors.append(_humanoid._info_err["hands_pose_err"])
        rewards.append(_humanoid._info_rew)
        body_rewards.append(_humanoid._info_rew["body_pose_reward"])
        hands_rewards.append(_humanoid._info_rew["hands_pose_reward"])
        body_vel_errors.append(_humanoid._info_err["body_vel_err"])
        hands_vel_errors.append(_humanoid._info_err["hands_vel_err"])
        body_vel_rewards.append(_humanoid._info_rew["body_vel_reward"])
        hands_vel_rewards.append(_humanoid._info_rew["hands_vel_reward"])

        _humanoid.initializePose(_humanoid._poseInterpolator, _humanoid._kin_model, initBase=True)

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

        _humanoid.computeAndApplyPDForces(desired_pose, maxForces=maxForces)

        _pybullet_client.stepSimulation()
        time.sleep(timeStep)

    print("Body error (sum):", sum(body_errors))
    print("Hands error (sum):", sum(hands_errors))
    print("Body rew (sum):", sum(body_rewards))
    print("Hands rew (sum):", sum(hands_rewards))
    print("Body error (avg):", sum(body_errors)/len(body_errors))
    print("Hands error (avg):", sum(hands_errors)/len(hands_errors))
    print("Body rew (avg):", sum(body_rewards)/len(body_rewards))
    print("Hands rew (avg):", sum(hands_rewards)/len(hands_rewards))

    print("Velocity")
    print("Body error (sum):", sum(body_vel_errors))
    print("Hands error (sum):", sum(hands_vel_errors))
    print("Body rew (sum):", sum(body_vel_rewards))
    print("Hands rew (sum):", sum(hands_vel_rewards))
    print("Body error (avg):", sum(body_vel_errors)/len(body_vel_errors))
    print("Hands error (avg):", sum(hands_vel_errors)/len(hands_vel_errors))
    print("Body rew (avg):", sum(body_vel_rewards)/len(body_vel_rewards))
    print("Hands rew (avg):", sum(hands_vel_rewards)/len(hands_vel_rewards))

    pose = sum(body_errors) + sum(hands_errors)

    _pybullet_client.disconnect()


if __name__ == '__main__':
    tune_controller()
