from pybullet_utils import bullet_client
import math
from deep_mimic.env.humanoid_pose_interpolator import HumanoidPoseInterpolator

class HumanoidPoseInterpolatorUpper(HumanoidPoseInterpolator):

    def __init__(self):
        super().__init__()
        pass

    def ConvertFromAction(self, pybullet_client, action):
        #turn action into pose

        self.Reset()  #?? needed?
        index = 0
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        self._chestRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
        #print("pose._chestRot=",pose._chestRot)

        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        self._neckRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

        """angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        self._rightHipRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index]
        index += 1
        self._rightKneeRot = [angle]

        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        self._rightAnkleRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)"""
        self._rightHipRot = [0, 0, 0, 1]
        self._rightKneeRot = [0]
        self._rightAnkleRot = [0, 0, 0, 1]

        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        self._rightShoulderRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index]
        index += 1
        self._rightElbowRot = [angle]

        """angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        self._leftHipRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index]
        index += 1
        self._leftKneeRot = [angle]

        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        self._leftAnkleRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)"""
        self._leftHipRot = [0, 0, 0, 1]
        self._leftKneeRot = [0]
        self._leftAnkleRot = [0, 0, 0, 1]

        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        self._leftShoulderRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index]
        index += 1
        self._leftElbowRot = [angle]

        pose = self.GetPose()
        return pose
