from pybullet_utils import pd_controller_stable
from deep_mimic.env import humanoid_pose_interpolator
from deep_mimic.env.humanoid_stable_pd_multiclip import HumanoidStablePDMultiClip
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


class HumanoidStablePDSelector(HumanoidStablePDMultiClip):

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

        self._poseInterpolator = humanoid_pose_interpolator.HumanoidPoseInterpolator()

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
            chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow, leftHip, leftKnee,
            leftAnkle, leftShoulder, leftElbow
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

        self._jointDofCounts = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]

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

    def resetPose(self):
        # print("resetPose with self._frame=", self._frame, " and self._frameFraction=",self._frameFraction)
        pose = self.computePose(self._frameFraction, self._current_clip)
        self.initializePose(self._poseInterpolator, self._sim_model, initBase=True)
        self.initializePose(self._poseInterpolator, self._kin_model, initBase=False)



    def calcCycleCount(self, simTime, cycleTime):
        return self._mocap_data.calcCycleCount(simTime, cycleTime)

    def getCycleTime(self, i):
        return self._mocap_data.getCycleTime(i)

    def setSimTime(self, t):
        self._simTime = t

        keyFrameDuration = self._mocap_data.getKeyFrameDuration(self._current_clip)
        cycleTime = self.getCycleTime(self._current_clip)

        self._cycleCount = self.calcCycleCount(t, cycleTime)

        frameTime = t - self._cycleCount * cycleTime
        if (frameTime < 0):
            frameTime += cycleTime

        self._frame = int(frameTime / keyFrameDuration)

        self._frameNext = self._frame + 1
        if (self._frameNext >= self._mocap_data.getNumFrames()):
            self._frameNext = self._frame

        self._frameFraction = (frameTime - self._frame * keyFrameDuration) / (keyFrameDuration)

    def computeCycleOffset(self, i):
        return self._mocap_data.computeCycleOffset(i)

    def computePose(self, frameFraction, i):
        frameData = self._mocap_data._motion_data[i]['Frames'][self._frame]
        frameDataNext = self._mocap_data._motion_data[i]['Frames'][self._frameNext]

        self._poseInterpolator.Slerp(frameFraction, frameData, frameDataNext, self._pybullet_client)

        self._cycleOffset = self.computeCycleOffset(i)
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

    def getState(self):

        stateVector = []
        phase = self.getPhase(self._current_clip)
        # print("phase=",phase)
        stateVector.append(phase)

        rootTransPos, rootTransOrn = self.buildOriginTrans()
        basePos, baseOrn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)

        rootPosRel, dummy = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                     basePos, [0, 0, 0, 1])

        localPos, localOrn = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                      basePos, baseOrn)

        localPos = [
            localPos[0] - rootPosRel[0], localPos[1] - rootPosRel[1], localPos[2] - rootPosRel[2]
        ]

        stateVector.append(rootPosRel[1])

        self.pb2dmJoints = range(15)

        linkIndicesSim = []
        for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
            linkIndicesSim.append(self.pb2dmJoints[pbJoint])

        linkStatesSim = self._pybullet_client.getLinkStates(self._sim_model, linkIndicesSim,
                                                            computeForwardKinematics=True, computeLinkVelocity=True)

        for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
            j = self.pb2dmJoints[pbJoint]
            # print("joint order:",j)
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

        mJointWeights = [
            0.20833, 0.10416, 0.0625, 0.10416, 0.0625, 0.041666666666666671, 0.0625, 0.0416, 0.00,
            0.10416, 0.0625, 0.0416, 0.0625, 0.0416, 0.0000
        ]

        num_end_effs = 0
        num_joints = 15

        root_rot_w = mJointWeights[root_id]
        rootPosSim, rootOrnSim = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
        rootPosKin, rootOrnKin = self._pybullet_client.getBasePositionAndOrientation(self._kin_model)
        linVelSim, angVelSim = self._pybullet_client.getBaseVelocity(self._sim_model)
        # don't read the velocities from the kinematic model (they are zero), use the pose interpolator velocity
        # see also issue https://github.com/bulletphysics/bullet3/issues/2401
        linVelKin = self._poseInterpolator._baseLinVel
        angVelKin = self._poseInterpolator._baseAngVel

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

        # print("pose_err=",pose_err)
        # print("vel_err=",vel_err)
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
        num_joints = 15
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

    def change_current_clip(self, current_clip):
        self._current_clip = current_clip
        # sync root pos and rot based on https://github.com/xbpeng/DeepMimic/issues/136
        root_pos_sim, root_orn_sim = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
        self._update_root_rot(root_orn_sim)
        self._update_root_pos(root_pos_sim)

    def _update_root_pos(self, root_pos_sim):
        self._mocap_data.translate_root_pos(self._current_clip, root_pos_sim)

    def _update_root_rot(self, root_orn_sim):
        ref_dir = [1, 0, 0]
        root_orn_clip = self._mocap_data._motion_data[self._current_clip]["Frames"][0][4:8]
        rot_vec_sim = self._pybullet_client.rotateVector(root_orn_sim, ref_dir)
        rot_vec_clip = self._pybullet_client.rotateVector(root_orn_clip, ref_dir)
        heading_sim = math.atan2(-rot_vec_sim[2], rot_vec_sim[0])
        heading_clip = math.atan2(-rot_vec_clip[2], rot_vec_clip[0])
        heading_offset = (heading_sim - heading_clip)
        self._mocap_data.translate_root_rot(self._current_clip, heading_offset)