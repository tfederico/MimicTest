from pybullet_utils import pd_controller_stable
from deep_mimic.env import simple_hand_pose_interpolator as hand_pose_interpolator
import math
import numpy as np

thumb_prox = 1
thumb_inter = 2
thumb_dist = 3
index_prox = 4
index_inter = 5
index_dist = 6
middle_prox = 7
middle_inter = 8
middle_dist = 9
ring_prox = 10
ring_inter = 11
ring_dist = 12
pinkie_prox = 13
pinkie_inter = 14
pinkie_dist = 15

jointFrictionForce = 0


class HandStablePD(object):

    def __init__(self, pybullet_client, mocap_data, timeStep,
                 useFixedBase=True, arg_parser=None, useComReward=False, kp=None, kd=None):

        self._pybullet_client = pybullet_client
        self._mocap_data = mocap_data
        self._arg_parser = arg_parser
        print("LOADING humanoid!")
        flags = self._pybullet_client.URDF_MAINTAIN_LINK_ORDER + self._pybullet_client.URDF_USE_SELF_COLLISION + self._pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self._sim_model = self._pybullet_client.loadURDF(
            "humanoid/new_simple_hand.urdf", [0, 0.5, 0],
            globalScaling=0.25,
            useFixedBase=useFixedBase,
            flags=flags)



        # self._pybullet_client.setCollisionFilterGroupMask(self._sim_model,-1,collisionFilterGroup=0,collisionFilterMask=0)
        # for j in range (self._pybullet_client.getNumJoints(self._sim_model)):
        #  self._pybullet_client.setCollisionFilterGroupMask(self._sim_model,j,collisionFilterGroup=0,collisionFilterMask=0)

        self._end_effectors = [thumb_dist, index_dist, middle_dist, ring_dist, pinkie_dist]

        self._kin_model = self._pybullet_client.loadURDF(
            "humanoid/new_simple_hand.urdf", [0, 0.5, 0],
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

        self._poseInterpolator = hand_pose_interpolator.HandPoseInterpolator()

        self._stablePD = pd_controller_stable.PDControllerStableMultiDof(self._pybullet_client)
        self._timeStep = timeStep

        if kp:
            self._kpOrg = [0] * 7 + [kp] * 15
        else:
            self._kpOrg = [0] * 7 + [0.64] * 15

        if kd:
            self._kdOrg = [0] * 7 + [kd] * 15
        else:
            self._kdOrg = [0] * 7 + [0.99] * 15

        self._jointIndicesAll = [
            thumb_prox, thumb_inter, thumb_dist,
            index_prox, index_inter, index_dist,
            middle_prox, middle_inter, middle_dist,
            ring_prox, ring_inter, ring_dist,
            pinkie_prox, pinkie_inter, pinkie_dist
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

        self._jointDofCounts = [1] * 15

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

    def initializePose(self, pose, phys_model, initBase):
   
        if initBase:
            self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose.base_pos,
                                                                  pose.base_orn)
            self._pybullet_client.resetBaseVelocity(phys_model, pose.base_lin_vel, pose.base_ang_vel)
        
        indices = self._jointIndicesAll
        jointPositions = [
            pose.thumb_prox, pose.thumb_inter, pose.thumb_dist,
            pose.index_prox, pose.index_inter, pose.index_dist,
            pose.middle_prox, pose.middle_inter, pose.middle_dist,
            pose.ring_prox, pose.ring_inter, pose.ring_dist,
            pose.pinkie_prox, pose.pinkie_inter, pose.pinkie_dist
        ]

        jointVelocities = [
            pose.thumb_prox_vel, pose.thumb_inter_vel, pose.thumb_dist_vel,
            pose.index_prox_vel, pose.index_inter_vel, pose.index_dist_vel,
            pose.middle_prox_vel, pose.middle_inter_vel, pose.middle_dist_vel,
            pose.ring_prox_vel, pose.ring_inter_vel, pose.ring_dist_vel,
            pose.pinkie_prox_vel, pose.pinkie_inter_vel, pose.pinkie_dist_vel
        ]
        self._pybullet_client.resetJointStatesMultiDof(phys_model, indices,
                                                       jointPositions, jointVelocities)



    def calcCycleCount(self, simTime, cycleTime):
        phases = simTime / cycleTime
        count = math.floor(phases)
        return count

    def getCycleTime(self):
        keyFrameDuration = self._mocap_data.KeyFrameDuraction()
        cycleTime = keyFrameDuration * (self._mocap_data.NumFrames() - 1)
        return cycleTime

    def setSimTime(self, t):
        self._simTime = t
        keyFrameDuration = self._mocap_data.KeyFrameDuraction()
        cycleTime = self.getCycleTime()
        self._cycleCount = self.calcCycleCount(t, cycleTime)
        frameTime = t - self._cycleCount * cycleTime
        if (frameTime < 0):
            frameTime += cycleTime
        self._frame = int(frameTime / keyFrameDuration)
        self._frameNext = self._frame + 1
        if (self._frameNext >= self._mocap_data.NumFrames()):
            self._frameNext = self._frame
        self._frameFraction = (frameTime - self._frame * keyFrameDuration) / (keyFrameDuration)

    def computeCycleOffset(self):
        lastFrame = self._mocap_data.NumFrames() - 1
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
        pose = self._poseInterpolator.Slerp(frameFraction, frameData, frameDataNext, self._pybullet_client)
        cycleOffset = self.computeCycleOffset()
        oldPos = self._poseInterpolator.base_pos
        self._poseInterpolator.base_pos = [
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
                targetPosition = [desiredPositions[dofIndex]]
                targetVelocity = [0]
            forces.append(force)
            targetPositions.append(targetPosition)
            targetVelocities.append(targetVelocity)
            dofIndex += self._jointDofCounts[index]

        self._pybullet_client.setJointMotorControlMultiDofArray(self._sim_model,
                                                                indices,
                                                                self._pybullet_client.POSITION_CONTROL,
                                                                targetPositions=targetPositions,
                                                                targetVelocities=targetVelocities,
                                                                forces=forces,
                                                                positionGains=kps,
                                                                velocityGains=kds)

    def getPhase(self):
        keyFrameDuration = self._mocap_data.KeyFrameDuraction()
        cycleTime = keyFrameDuration * (self._mocap_data.NumFrames() - 1)
        phase = self._simTime / cycleTime
        phase = math.fmod(phase, 1.0)
        if (phase < 0):
            phase += 1
        return phase

    def buildHeadingTrans(self, rootOrn):
        eul = self._pybullet_client.getEulerFromQuaternion(rootOrn)
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

        stateVector.append(rootPosRel[1])

        self.pb2dmJoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        linkIndicesSim = []
        for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
            linkIndicesSim.append(self.pb2dmJoints[pbJoint])

        linkStatesSim = self._pybullet_client.getLinkStates(self._sim_model, linkIndicesSim,
                                                            computeForwardKinematics=True, computeLinkVelocity=True)

        for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
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

    def getReward(self, pose):
        """Compute and return the pose-based reward."""
        # from DeepMimic double cSceneImitate::CalcRewardImitate
        # todo: compensate for ground height in some parts, once we move to non-flat terrain
        # not values from the paper, but from the published code.
        pose_w = 0.5
        vel_w = 0.05
        end_eff_w = 0.15
        # does not exist in paper
        root_w = 0.2
        if self._useComReward:
            com_w = 0.1
        else:
            com_w = 0

        total_w = pose_w + vel_w + end_eff_w + root_w + com_w
        pose_w /= total_w
        vel_w /= total_w
        end_eff_w /= total_w
        root_w /= total_w
        com_w /= total_w

        pose_scale = 2
        vel_scale = 0.1
        end_eff_scale = 40
        root_scale = 5
        com_scale = 10
        err_scale = 1  # error scale

        reward = 0

        pose_err = 0
        vel_err = 0
        end_eff_err = 0
        root_err = 0
        com_err = 0
        heading_err = 0

        # create a mimic reward, comparing the dynamics humanoid with a kinematic one

        if self._useComReward:
            comSim, comSimVel = self.computeCOMposVel(self._sim_model)
            comKin, comKinVel = self.computeCOMposVel(self._kin_model)

        root_id = 0

        num_end_effs = 5
        num_joints = 16

        mJointWeights = [1]*num_joints # TODO: replace values

        root_rot_w = mJointWeights[root_id]
        rootPosSim, rootOrnSim = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
        rootPosKin, rootOrnKin = self._pybullet_client.getBasePositionAndOrientation(self._kin_model)
        linVelSim, angVelSim = self._pybullet_client.getBaseVelocity(self._sim_model)
        # don't read the velocities from the kinematic model (they are zero), use the pose interpolator velocity
        # see also issue https://github.com/bulletphysics/bullet3/issues/2401
        linVelKin = self._poseInterpolator.base_lin_vel
        angVelKin = self._poseInterpolator.base_ang_vel

        root_rot_err = self.calcRootRotDiff(rootOrnSim, rootOrnKin)
        pose_err += root_rot_w * root_rot_err

        root_vel_diff = [
            linVelSim[0] - linVelKin[0], linVelSim[1] - linVelKin[1], linVelSim[2] - linVelKin[2]
        ]
        root_vel_err = root_vel_diff[0] * root_vel_diff[0] + root_vel_diff[1] * root_vel_diff[
            1] + root_vel_diff[2] * root_vel_diff[2]

        root_ang_vel_err = self.calcRootAngVelErr(angVelSim, angVelKin)
        vel_err += root_rot_w * root_ang_vel_err

        jointIndices = range(num_joints)
        simJointStates = self._pybullet_client.getJointStatesMultiDof(self._sim_model, jointIndices)
        kinJointStates = self._pybullet_client.getJointStatesMultiDof(self._kin_model, jointIndices)
        linkStatesSim = self._pybullet_client.getLinkStates(self._sim_model, jointIndices)
        linkStatesKin = self._pybullet_client.getLinkStates(self._kin_model, jointIndices)
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

            is_end_eff = j in self._end_effectors

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

        if (num_end_effs > 0):
            end_eff_err /= num_end_effs

        root_pos_diff = [
            rootPosSim[0] - rootPosKin[0], rootPosSim[1] - rootPosKin[1], rootPosSim[2] - rootPosKin[2]
        ]
        root_pos_err = root_pos_diff[0] * root_pos_diff[0] + root_pos_diff[1] * root_pos_diff[
            1] + root_pos_diff[2] * root_pos_diff[2]

        root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_ang_vel_err

        # COM error in initial code -> COM velocities
        if self._useComReward:
            com_err = 0.1 * np.sum(np.square(comKinVel - comSimVel))
        # com_err = 0.1 * np.sum(np.square(comKin - comSim))
        # com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm()

        pose_reward = math.exp(-err_scale * pose_scale * pose_err)
        vel_reward = math.exp(-err_scale * vel_scale * vel_err)
        end_eff_reward = math.exp(-err_scale * end_eff_scale * end_eff_err)
        root_reward = math.exp(-err_scale * root_scale * root_err)
        com_reward = math.exp(-err_scale * com_scale * com_err)

        reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward + root_w * root_reward + com_w * com_reward

        info_rew = dict(
            pose_reward=pose_reward,
            vel_reward=vel_reward,
            end_eff_reward=end_eff_reward,
            root_reward=root_reward,
            imitation_reward=reward
        )

        info_errs = dict(
            pose_err=pose_err,
            vel_err=vel_err,
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
        num_joints = 16
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



def tune_controller(args):
    import pybullet as p1
    from pybullet_utils import bullet_client
    import data as pybullet_data
    from pybullet_utils.arg_parser import ArgParser
    from deep_mimic.env.pybullet_deep_mimic_env_hand import InitializationStrategy
    from deep_mimic.env import motion_capture_data
    import time
    import wandb

    wandb.init(config=args)
    args = wandb.config

    arg_file = "run_humanoid3d_signer_args.txt"
    arg_parser = ArgParser()
    path = pybullet_data.getDataPath() + "/args/" + arg_file
    succ = arg_parser.load_file(path)
    timeStep = 1. / 30
    _init_strategy = InitializationStrategy.START
    _pybullet_client = bullet_client.BulletClient(connection_mode=p1.GUI)
    # # disable 'GUI' since it slows down a lot on Mac OSX and some other platforms
    _pybullet_client.configureDebugVisualizer(_pybullet_client.COV_ENABLE_GUI, 0)
    _pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    z2y = _pybullet_client.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
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
    useFixedBase = True

    _humanoid = HandStablePD(_pybullet_client, _mocapData, timeStep, useFixedBase, arg_parser, kp=args.kp, kd=args.kd)

    _isInitialized = True
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

    n_joints = _pybullet_client.getNumJoints(_humanoid._sim_model)

    cycle = _mocapData.getCycleTime()

    action = _mocapData._motion_data['Frames'][_humanoid._frameNext]
    _humanoid.convertActionToPose(action[7:])

    kin_joint = []
    sim_joint = []
    kin_vel = []
    sim_vel = []

    steps = range(1200)

    for _ in steps:
        _pybullet_client.setTimeStep(timeStep)
        _humanoid._timeStep = timeStep
        t += timeStep
        _humanoid.setSimTime(t)
        kinPose = _humanoid.computePose(_humanoid._frameFraction)
        _humanoid.initializePose(_humanoid._poseInterpolator, _humanoid._kin_model, initBase=True)
        maxForces = [0] * 7 + [500] * 15

        _humanoid.computeAndApplyPDForces(kinPose, maxForces=maxForces)
        state = _pybullet_client.getJointStates(_humanoid._sim_model, list(range(16)))

        _pybullet_client.stepSimulation()
        # _humanoid._sim_model
        state = _pybullet_client.getJointStates(_humanoid._sim_model, list(range(16)))
        simPose = [s[0] for s in state]
        simPose = [0.0, 0.9, 0.0, 1, 0, 0, 0] + simPose[1:]
        kinVelocities = _humanoid._poseInterpolator.GetVelocities()
        simVelocities = [s[1] for s in state]

        kin_joint.append(kinPose[13])
        sim_joint.append(simPose[13])

        kin_vel.append(kinVelocities[6])
        sim_vel.append(simVelocities[6])

        time.sleep(1/100)

    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(4)
    #
    # ax[0].plot(list(steps), kin_joint, color="blue")
    # ax[1].plot(list(steps), sim_joint, color="red")
    pos_err = [k - s for k, s in zip(kin_joint, sim_joint)]
    # ax[0].plot(list(steps), pos_err, color="green")

    abs_pos_err = [abs(p) for p in pos_err]
    # ax[2].plot(list(steps), kin_vel, color="blue")
    # ax[3].plot(list(steps), sim_vel, color="red")
    vel_err = [k - s for k, s in zip(kin_vel, sim_vel)]
    # ax[1].plot(list(steps), vel_err, color="green")

    abs_vel_err = [abs(v) for v in vel_err]
    # plt.show()

    log = {
        "pose": sum(abs_pos_err),
        "velocity": sum(abs_vel_err),
        "error": sum(abs_pos_err) + sum(abs_vel_err)
    }

    wandb.log(log)

    _pybullet_client.disconnect()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--kp', type=float)
    parser.add_argument('--kd', type=float)

    args = parser.parse_args()
    tune_controller(args)