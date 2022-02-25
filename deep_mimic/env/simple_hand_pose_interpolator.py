from pybullet_utils import bullet_client
import math


class HandPoseInterpolator(object):

    def __init__(self):
        pass

    def Reset(self,
              base_pos=[0, 0, 0],
              base_orn=[0, 0, 0, 1],
              thumb_prox=[0],
              thumb_inter=[0],
              thumb_dist=[0],
              index_prox=[0],
              index_inter=[0],
              index_dist=[0],
              middle_prox=[0],
              middle_inter=[0],
              middle_dist=[0],
              ring_prox=[0],
              ring_inter=[0],
              ring_dist=[0],
              pinkie_prox=[0],
              pinkie_inter=[0],
              pinkie_dist=[0],
              base_lin_vel=[0],
              base_ang_vel=[0],
              thumb_prox_vel=[0],
              thumb_inter_vel=[0],
              thumb_dist_vel=[0],
              index_prox_vel=[0],
              index_inter_vel=[0],
              index_dist_vel=[0],
              middle_prox_vel=[0],
              middle_inter_vel=[0],
              middle_dist_vel=[0],
              ring_prox_vel=[0],
              ring_inter_vel=[0],
              ring_dist_vel=[0],
              pinkie_prox_vel=[0],
              pinkie_inter_vel=[0],
              pinkie_dist_vel=[0]):

        self.base_pos = base_pos
        self.base_orn = base_orn
        self.thumb_prox = thumb_prox
        self.thumb_inter = thumb_inter
        self.thumb_dist = thumb_dist
        self.index_prox = index_prox
        self.index_inter = index_inter
        self.index_dist = index_dist
        self.middle_prox = middle_prox
        self.middle_inter = middle_inter
        self.middle_dist = middle_dist
        self.ring_prox = ring_prox
        self.ring_inter = ring_inter
        self.ring_dist = ring_dist
        self.pinkie_prox = pinkie_prox
        self.pinkie_inter = pinkie_inter
        self.pinkie_dist = pinkie_dist
        self.base_lin_vel = base_lin_vel
        self.base_ang_vel = base_ang_vel
        self.thumb_prox_vel = thumb_prox_vel
        self.thumb_inter_vel = thumb_inter_vel
        self.thumb_dist_vel = thumb_dist_vel
        self.index_prox_vel = index_prox_vel
        self.index_inter_vel = index_inter_vel
        self.index_dist_vel = index_dist_vel
        self.middle_prox_vel = middle_prox_vel
        self.middle_inter_vel = middle_inter_vel
        self.middle_dist_vel = middle_dist_vel
        self.ring_prox_vel = ring_prox_vel
        self.ring_inter_vel = ring_inter_vel
        self.ring_dist_vel = ring_dist_vel
        self.pinkie_prox_vel = pinkie_prox_vel
        self.pinkie_inter_vel = pinkie_inter_vel
        self.pinkie_dist_vel = pinkie_dist_vel


    def ComputeLinVel(self, posStart, posEnd, deltaTime):
        vel = [(posEnd[0] - posStart[0]) / deltaTime, (posEnd[1] - posStart[1]) / deltaTime,
               (posEnd[2] - posStart[2]) / deltaTime]
        return vel

    def ComputeAngVel(self, ornStart, ornEnd, deltaTime, bullet_client):
        dorn = bullet_client.getDifferenceQuaternion(ornStart, ornEnd)
        axis, angle = bullet_client.getAxisAngleFromQuaternion(dorn)
        angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
                  (axis[2] * angle) / deltaTime]
        return angVel

    def ComputeAngVelRel(self, ornStart, ornEnd, deltaTime, bullet_client):
        ornStartConjugate = [-ornStart[0], -ornStart[1], -ornStart[2], ornStart[3]]
        pos_diff, q_diff = bullet_client.multiplyTransforms([0, 0, 0], ornStartConjugate, [0, 0, 0],
                                                            ornEnd)
        axis, angle = bullet_client.getAxisAngleFromQuaternion(q_diff)
        angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
                  (axis[2] * angle) / deltaTime]
        return angVel

    def NormalizeQuaternion(self, orn):
        length2 = orn[0] * orn[0] + orn[1] * orn[1] + orn[2] * orn[2] + orn[3] * orn[3]
        if (length2 > 0):
            length = math.sqrt(length2)
            orn[0] /= length
            orn[1] /= length
            orn[2] /= length
            orn[3] /= length
            return orn

    def GetVelocities(self):
        vel = [
            *self.thumb_prox_vel,
            *self.thumb_inter_vel,
            *self.thumb_dist_vel,
            *self.index_prox_vel,
            *self.index_inter_vel,
            *self.index_dist_vel,
            *self.middle_prox_vel,
            *self.middle_inter_vel,
            *self.middle_dist_vel,
            *self.ring_prox_vel,
            *self.ring_inter_vel,
            *self.ring_dist_vel,
            *self.pinkie_prox_vel,
            *self.pinkie_inter_vel,
            *self.pinkie_dist_vel
        ]
        return vel

    def GetPose(self):
        pose = [
            *self.base_pos,
            *self.base_orn,
            *self.thumb_prox,
            *self.thumb_inter,
            *self.thumb_dist,
            *self.index_prox,
            *self.index_inter,
            *self.index_dist,
            *self.middle_prox,
            *self.middle_inter,
            *self.middle_dist,
            *self.ring_prox,
            *self.ring_inter,
            *self.ring_dist,
            *self.pinkie_prox,
            *self.pinkie_inter,
            *self.pinkie_dist
        ]
        return pose

    def Slerp(self, frameFraction, frameData, frameDataNext, bullet_client):

        i = 0
        keyFrameDuration = frameData[i]
        basePos1Start = [frameData[i+1], frameData[i+2], frameData[i+3]]
        basePos1End = [frameDataNext[i+1], frameDataNext[i+2], frameDataNext[i+3]]
        self.base_pos = [
            basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
            basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
            basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
        ]

        i += 4
        self.base_lin_vel = self.ComputeLinVel(basePos1Start, basePos1End, keyFrameDuration)
        baseOrn1Start = [frameData[i+1], frameData[i+2], frameData[i+3], frameData[i]]
        baseOrn1Next = [frameDataNext[i+1], frameDataNext[i+2], frameDataNext[i+3], frameDataNext[i]]
        self.base_orn = bullet_client.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
        self.base_ang_vel = self.ComputeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration, bullet_client)

        i += 4
        thumb_prox_start = [frameData[i]]
        thumb_prox_end = [frameDataNext[i]]
        self.thumb_prox = [thumb_prox_start[0] + frameFraction * (thumb_prox_end[0] - thumb_prox_start[0])]
        self.thumb_prox_vel = [(thumb_prox_end[0] - thumb_prox_start[0]) / keyFrameDuration]

        i += 1
        thumb_inter_start = [frameData[i]]
        thumb_inter_end = [frameDataNext[i]]
        self.thumb_inter = [thumb_inter_start[0] + frameFraction * (thumb_inter_end[0] - thumb_inter_start[0])]
        self.thumb_inter_vel = [(thumb_inter_end[0] - thumb_inter_start[0]) / keyFrameDuration]

        i += 1
        thumb_dist_start = [frameData[i]]
        thumb_dist_end = [frameDataNext[i]]
        self.thumb_dist = [thumb_dist_start[0] + frameFraction * (thumb_dist_end[0] - thumb_dist_start[0])]
        self.thumb_dist_vel = [(thumb_inter_end[0] - thumb_inter_start[0]) / keyFrameDuration]

        i += 1
        index_prox_start = [frameData[i]]
        index_prox_end = [frameDataNext[i]]
        self.index_prox = [index_prox_start[0] + frameFraction * (index_prox_end[0] - index_prox_start[0])]
        self.index_prox_vel = [(index_prox_end[0] - index_prox_start[0]) / keyFrameDuration]

        i += 1
        index_inter_start = [frameData[i]]
        index_inter_end = [frameDataNext[i]]
        self.index_inter = [index_inter_start[0] + frameFraction * (index_inter_end[0] - index_inter_start[0])]
        self.index_inter_vel = [(index_inter_end[0] - index_inter_start[0]) / keyFrameDuration]

        i += 1
        index_dist_start = [frameData[i]]
        index_dist_end = [frameDataNext[i]]
        self.index_dist = [index_dist_start[0] + frameFraction * (index_dist_end[0] - index_dist_start[0])]
        self.index_dist_vel = [(index_dist_end[0] - index_dist_start[0]) / keyFrameDuration]

        i += 1
        middle_prox_start = [frameData[i]]
        middle_prox_end = [frameDataNext[i]]
        self.middle_prox = [middle_prox_start[0] + frameFraction * (middle_prox_end[0] - middle_prox_start[0])]
        self.middle_prox_vel = [(middle_prox_end[0] - middle_prox_start[0]) / keyFrameDuration]

        i += 1
        middle_inter_start = [frameData[i]]
        middle_inter_end = [frameDataNext[i]]
        self.middle_inter = [middle_inter_start[0] + frameFraction * (middle_inter_end[0] - middle_inter_start[0])]
        self.middle_inter_vel = [(middle_inter_end[0] - middle_inter_start[0]) / keyFrameDuration]

        i += 1
        middle_dist_start = [frameData[i]]
        middle_dist_end = [frameDataNext[i]]
        self.middle_dist = [middle_dist_start[0] + frameFraction * (middle_dist_end[0] - middle_dist_start[0])]
        self.middle_dist_vel = [(middle_dist_end[0] - middle_dist_start[0]) / keyFrameDuration]

        i += 1
        ring_prox_start = [frameData[i]]
        ring_prox_end = [frameDataNext[i]]
        self.ring_prox = [ring_prox_start[0] + frameFraction * (ring_prox_end[0] - ring_prox_start[0])]
        self.ring_prox_vel = [(ring_prox_end[0] - ring_prox_start[0]) / keyFrameDuration]

        i += 1
        ring_inter_start = [frameData[i]]
        ring_inter_end = [frameDataNext[i]]
        self.ring_inter = [ring_inter_start[0] + frameFraction * (ring_inter_end[0] - ring_inter_start[0])]
        self.ring_inter_vel = [(ring_inter_end[0] - ring_inter_start[0]) / keyFrameDuration]

        i += 1
        ring_dist_start = [frameData[i]]
        ring_dist_end = [frameDataNext[i]]
        self.ring_dist = [ring_dist_start[0] + frameFraction * (ring_dist_end[0] - ring_dist_start[0])]
        self.ring_dist_vel = [(ring_dist_end[0] - ring_dist_start[0]) / keyFrameDuration]

        i += 1
        pinkie_prox_start = [frameData[i]]
        pinkie_prox_end = [frameDataNext[i]]
        self.pinkie_prox = [pinkie_prox_start[0] + frameFraction * (pinkie_prox_end[0] - pinkie_prox_start[0])]
        self.pinkie_prox_vel = [(pinkie_prox_end[0] - pinkie_prox_start[0]) / keyFrameDuration]

        i += 1
        pinkie_inter_start = [frameData[i]]
        pinkie_inter_end = [frameDataNext[i]]
        self.pinkie_inter = [pinkie_inter_start[0] + frameFraction * (pinkie_inter_end[0] - pinkie_inter_start[0])]
        self.pinkie_inter_vel = [(pinkie_inter_end[0] - pinkie_inter_start[0]) / keyFrameDuration]

        i += 1
        pinkie_dist_start = [frameData[i]]
        pinkie_dist_end = [frameDataNext[i]]
        self.pinkie_dist = [pinkie_dist_start[0] + frameFraction * (pinkie_dist_end[0] - pinkie_dist_start[0])]
        self.pinkie_dist_vel = [(pinkie_dist_end[0] - pinkie_dist_start[0]) / keyFrameDuration]

        pose = self.GetPose()
        return pose

    def ConvertFromAction(self, pybullet_client, action):
        #turn action into pose

        self.Reset()  #?? needed?
        index = 0
        angle = action[index]
        index += 1
        self.thumb_prox = [angle]
        angle = action[index]
        index += 1
        self.thumb_inter = [angle]
        angle = action[index]
        index += 1
        self.thumb_dist = [angle]

        angle = action[index]
        index += 1
        self.index_prox = [angle]
        angle = action[index]
        index += 1
        self.index_inter = [angle]
        angle = action[index]
        index += 1
        self.index_dist = [angle]

        angle = action[index]
        index += 1
        self.middle_prox = [angle]
        angle = action[index]
        index += 1
        self.middle_inter = [angle]
        angle = action[index]
        index += 1
        self.middle_dist = [angle]

        angle = action[index]
        index += 1
        self.ring_prox = [angle]
        angle = action[index]
        index += 1
        self.ring_inter = [angle]
        angle = action[index]
        index += 1
        self.ring_dist = [angle]

        angle = action[index]
        index += 1
        self.pinkie_prox = [angle]
        angle = action[index]
        index += 1
        self.pinkie_inter = [angle]
        angle = action[index]
        index += 1
        self.pinkie_dist = [angle]

        pose = self.GetPose()
        return pose
