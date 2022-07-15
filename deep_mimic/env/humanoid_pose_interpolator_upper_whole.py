import math


class WholeHumanoidPoseInterpolator(object):

    def __init__(self):
        pass

    def Reset(self,
                basePos=[0, 0, 0],
                baseOrn=[0, 0, 0, 1],
                chestRot=[0, 0, 0, 1],
                neckRot=[0, 0, 0, 1],
                rightHipRot=[0, 0, 0, 1],
                rightKneeRot=[0],
                rightAnkleRot=[0, 0, 0, 1],
                rightShoulderRot=[0, 0, 0, 1],
                rightElbowRot=[0],
                rightWristRot=[0],
                rightThumbProx=[0],
                rightThumbInter=[0],
                rightThumbDist=[0],
                rightIndexProx=[0], rightIndexInter=[0], rightIndexDist=[0],
                rightMiddleProx=[0], rightMiddleInter=[0], rightMiddleDist=[0],
                rightRingProx=[0], rightRingInter=[0], rightRingDist=[0],
                rightPinkieProx=[0], rightPinkieInter=[0], rightPinkieDist=[0],
                leftHipRot=[0, 0, 0, 1],
                leftKneeRot=[0],
                leftAnkleRot=[0, 0, 0, 1],
                leftShoulderRot=[0, 0, 0, 1],
                leftElbowRot=[0],
                leftWristRot=[0],
                leftThumbProx=[0], leftThumbInter=[0], leftThumbDist=[0],
                leftIndexProx=[0], leftIndexInter=[0], leftIndexDist=[0],
                leftMiddleProx=[0], leftMiddleInter=[0], leftMiddleDist=[0],
                leftRingProx=[0], leftRingInter=[0], leftRingDist=[0],
                leftPinkieProx=[0], leftPinkieInter=[0], leftPinkieDist=[0],
                baseLinVel=[0, 0, 0],
                baseAngVel=[0, 0, 0],
                chestVel=[0, 0, 0],
                neckVel=[0, 0, 0],
                rightHipVel=[0, 0, 0],
                rightKneeVel=[0],
                rightAnkleVel=[0, 0, 0],
                rightShoulderVel=[0, 0, 0],
                rightElbowVel=[0],
                rightWristVel=[0],
                rightThumbProxVel=[0], rightThumbInterVel=[0], rightThumbDistVel=[0],
                rightIndexProxVel=[0], rightIndexInterVel=[0], rightIndexDistVel=[0],
                rightMiddleProxVel=[0], rightMiddleInterVel=[0], rightMiddleDistVel=[0],
                rightRingProxVel=[0], rightRingInterVel=[0], rightRingDistVel=[0],
                rightPinkieProxVel=[0], rightPinkieInterVel=[0], rightPinkieDistVel=[0],
                leftHipVel=[0, 0, 0],
                leftKneeVel=[0],
                leftAnkleVel=[0, 0, 0],
                leftShoulderVel=[0, 0, 0],
                leftElbowVel=[0],
                leftWristVel=[0],
                leftThumbProxVel=[0], leftThumbInterVel=[0], leftThumbDistVel=[0],
                leftIndexProxVel=[0], leftIndexInterVel=[0], leftIndexDistVel=[0],
                leftMiddleProxVel=[0], leftMiddleInterVel=[0], leftMiddleDistVel=[0],
                leftRingProxVel=[0], leftRingInterVel=[0], leftRingDistVel=[0],
                leftPinkieProxVel=[0], leftPinkieInterVel=[0], leftPinkieDistVel=[0]
              ):

        self._basePos = basePos
        self._baseLinVel = baseLinVel
        #print("HumanoidPoseInterpolator.Reset: baseLinVel = ", baseLinVel)
        self._baseOrn = baseOrn
        self._baseAngVel = baseAngVel

        self._chestRot = chestRot
        self._chestVel = chestVel
        self._neckRot = neckRot
        self._neckVel = neckVel
        self._rightHipRot = rightHipRot
        self._rightHipVel = rightHipVel
        self._rightKneeRot = rightKneeRot
        self._rightKneeVel = rightKneeVel
        self._rightAnkleRot = rightAnkleRot
        self._rightAnkleVel = rightAnkleVel
        self._rightShoulderRot = rightShoulderRot
        self._rightShoulderVel = rightShoulderVel
        self._rightElbowRot = rightElbowRot
        self._rightElbowVel = rightElbowVel
        self._rightWristRot = rightWristRot
        self._rightWristVel = rightWristVel
        self._rightThumbProx = rightThumbProx
        self._rightThumbInter = rightThumbInter
        self._rightThumbDist = rightThumbDist
        self._rightIndexProx = rightIndexProx
        self._rightIndexInter = rightIndexInter
        self._rightIndexDist = rightIndexDist
        self._rightMiddleProx = rightMiddleProx
        self._rightMiddleInter = rightMiddleInter
        self._rightMiddleDist = rightMiddleDist
        self._rightRingProx = rightRingProx
        self._rightRingInter = rightRingInter
        self._rightRingDist = rightRingDist
        self._rightPinkieProx = rightPinkieProx
        self._rightPinkieInter = rightPinkieInter
        self._rightPinkieDist = rightPinkieDist
        self._leftHipRot = leftHipRot
        self._leftHipVel = leftHipVel
        self._leftKneeRot = leftKneeRot
        self._leftKneeVel = leftKneeVel
        self._leftAnkleRot = leftAnkleRot
        self._leftAnkleVel = leftAnkleVel
        self._leftShoulderRot = leftShoulderRot
        self._leftShoulderVel = leftShoulderVel
        self._leftElbowRot = leftElbowRot
        self._leftElbowVel = leftElbowVel
        self._leftWristRot = leftWristRot
        self._leftWristVel = leftWristVel
        self._leftThumbProx = leftThumbProx
        self._leftThumbInter = leftThumbInter
        self._leftThumbDist = leftThumbDist
        self._leftIndexProx = leftIndexProx
        self._leftIndexInter = leftIndexInter
        self._leftIndexDist = leftIndexDist
        self._leftMiddleProx = leftMiddleProx
        self._leftMiddleInter = leftMiddleInter
        self._leftMiddleDist = leftMiddleDist
        self._leftRingProx = leftRingProx
        self._leftRingInter = leftRingInter
        self._leftRingDist = leftRingDist
        self._leftPinkieProx = leftPinkieProx
        self._leftPinkieInter = leftPinkieInter
        self._leftPinkieDist = leftPinkieDist
        self._rightThumbDistVel = rightThumbDistVel
        self._rightThumbInterVel = rightThumbInterVel
        self._rightThumbProxVel = rightThumbProxVel
        self._rightIndexDistVel = rightIndexDistVel
        self._rightIndexInterVel = rightIndexInterVel
        self._rightIndexProxVel = rightIndexProxVel
        self._rightMiddleDistVel = rightMiddleDistVel
        self._rightMiddleInterVel = rightMiddleInterVel
        self._rightMiddleProxVel = rightMiddleProxVel
        self._rightRingDistVel = rightRingDistVel
        self._rightRingInterVel = rightRingInterVel
        self._rightRingProxVel = rightRingProxVel
        self._rightPinkieDistVel = rightPinkieDistVel
        self._rightPinkieInterVel = rightPinkieInterVel
        self._rightPinkieProxVel = rightPinkieProxVel
        self._leftThumbDistVel = leftThumbDistVel
        self._leftThumbInterVel = leftThumbInterVel
        self._leftThumbProxVel = leftThumbProxVel
        self._leftIndexDistVel = leftIndexDistVel
        self._leftIndexInterVel = leftIndexInterVel
        self._leftIndexProxVel = leftIndexProxVel
        self._leftMiddleDistVel = leftMiddleDistVel
        self._leftMiddleInterVel = leftMiddleInterVel
        self._leftMiddleProxVel = leftMiddleProxVel
        self._leftRingDistVel = leftRingDistVel
        self._leftRingInterVel = leftRingInterVel
        self._leftRingProxVel = leftRingProxVel
        self._leftPinkieDistVel = leftPinkieDistVel
        self._leftPinkieInterVel = leftPinkieInterVel
        self._leftPinkieProxVel = leftPinkieProxVel


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


    def GetPose(self):
        pose = [
            *self._basePos,
            *self._baseOrn,
            *self._chestRot,
            *self._neckRot,
            *self._rightHipRot,
            *self._rightKneeRot,
            *self._rightAnkleRot,
            *self._rightShoulderRot,
            *self._rightElbowRot,
            *self._rightWristRot,
            *self._rightThumbProx, *self._rightThumbInter, *self._rightThumbDist,
            *self._rightIndexProx, *self._rightIndexInter, *self._rightIndexDist,
            *self._rightMiddleProx, *self._rightMiddleInter, *self._rightMiddleDist,
            *self._rightRingProx, *self._rightRingInter, *self._rightRingDist,
            *self._rightPinkieProx, *self._rightPinkieInter, *self._rightPinkieDist,
            *self._leftHipRot,
            *self._leftKneeRot,
            *self._leftAnkleRot,
            *self._leftShoulderRot,
            *self._leftElbowRot,
            *self._leftWristRot,
            *self._leftThumbProx, *self._leftThumbInter, *self._leftThumbDist,
            *self._leftIndexProx, *self._leftIndexInter, *self._leftIndexDist,
            *self._leftMiddleProx, *self._leftMiddleInter, *self._leftMiddleDist,
            *self._leftRingProx, *self._leftRingInter, *self._leftRingDist,
            *self._leftPinkieProx, *self._leftPinkieInter, *self._leftPinkieDist
        ]
        return pose

    def GetVelocities(self):
        vel = [
            *self._baseLinVel, *self._chestVel, *self._neckVel, *self._rightHipVel, *self._rightKneeVel,
            *self._rightAnkleVel, *self._rightShoulderVel, *self._rightElbowVel, *self._rightWristVel,
            *self._rightThumbProxVel, *self._rightThumbInterVel, *self._rightThumbDistVel,
            *self._rightIndexProxVel, *self._rightIndexInterVel, *self._rightIndexDistVel,
            *self._rightMiddleProxVel, *self._rightMiddleInterVel, *self._rightMiddleDistVel,
            *self._rightRingProxVel, *self._rightRingInterVel, *self._rightRingDistVel,
            *self._rightPinkieProxVel, *self._rightPinkieInterVel, *self._rightPinkieDistVel,
            *self._leftHipVel, *self._leftKneeVel, *self._leftAnkleVel, *self._leftShoulderVel,
            *self._leftElbowVel, *self._leftWristVel, *self._leftThumbProxVel, *self._leftThumbInterVel,
            *self._leftThumbDistVel, *self._leftIndexProxVel, *self._leftIndexInterVel, *self._leftIndexDistVel,
            *self._leftMiddleProxVel, *self._leftMiddleInterVel, *self._leftMiddleDistVel, *self._leftRingProxVel,
            *self._leftRingInterVel, *self._leftRingDistVel, *self._leftPinkieProxVel, *self._leftPinkieInterVel,
            *self._leftPinkieDistVel
        ]
        return vel

    def Slerp(self, frameFraction, frameData, frameDataNext, bullet_client):
        keyFrameDuration = frameData[0]
        basePos1Start = [frameData[1], frameData[2], frameData[3]]
        basePos1End = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
        self._basePos = [
            basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
            basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
            basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
        ]
        self._baseLinVel = self.ComputeLinVel(basePos1Start, basePos1End, keyFrameDuration)
        baseOrn1Start = [frameData[5], frameData[6], frameData[7], frameData[4]]
        baseOrn1Next = [frameDataNext[5], frameDataNext[6], frameDataNext[7], frameDataNext[4]]
        self._baseOrn = bullet_client.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
        self._baseAngVel = self.ComputeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration,
                                              bullet_client)

        chestRotStart = [frameData[9], frameData[10], frameData[11], frameData[8]]
        chestRotEnd = [frameDataNext[9], frameDataNext[10], frameDataNext[11], frameDataNext[8]]
        self._chestRot = bullet_client.getQuaternionSlerp(chestRotStart, chestRotEnd, frameFraction)
        self._chestVel = self.ComputeAngVelRel(chestRotStart, chestRotEnd, keyFrameDuration,
                                               bullet_client)

        neckRotStart = [frameData[13], frameData[14], frameData[15], frameData[12]]
        neckRotEnd = [frameDataNext[13], frameDataNext[14], frameDataNext[15], frameDataNext[12]]
        self._neckRot = bullet_client.getQuaternionSlerp(neckRotStart, neckRotEnd, frameFraction)
        self._neckVel = self.ComputeAngVelRel(neckRotStart, neckRotEnd, keyFrameDuration,
                                              bullet_client)

        rightHipRotStart = [frameData[17], frameData[18], frameData[19], frameData[16]]
        rightHipRotEnd = [frameDataNext[17], frameDataNext[18], frameDataNext[19], frameDataNext[16]]
        self._rightHipRot = bullet_client.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd,
                                                             frameFraction)
        self._rightHipVel = self.ComputeAngVelRel(rightHipRotStart, rightHipRotEnd, keyFrameDuration,
                                                  bullet_client)

        rightKneeRotStart = [frameData[20]]
        rightKneeRotEnd = [frameDataNext[20]]
        self._rightKneeRot = [
            rightKneeRotStart[0] + frameFraction * (rightKneeRotEnd[0] - rightKneeRotStart[0])
        ]
        self._rightKneeVel = [(rightKneeRotEnd[0] - rightKneeRotStart[0]) / keyFrameDuration]

        rightAnkleRotStart = [frameData[22], frameData[23], frameData[24], frameData[21]]
        rightAnkleRotEnd = [frameDataNext[22], frameDataNext[23], frameDataNext[24], frameDataNext[21]]
        self._rightAnkleRot = bullet_client.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd,
                                                               frameFraction)
        self._rightAnkleVel = self.ComputeAngVelRel(rightAnkleRotStart, rightAnkleRotEnd,
                                                    keyFrameDuration, bullet_client)

        rightShoulderRotStart = [frameData[26], frameData[27], frameData[28], frameData[25]]
        rightShoulderRotEnd = [
            frameDataNext[26], frameDataNext[27], frameDataNext[28], frameDataNext[25]
        ]
        self._rightShoulderRot = bullet_client.getQuaternionSlerp(rightShoulderRotStart,
                                                                  rightShoulderRotEnd, frameFraction)
        self._rightShoulderVel = self.ComputeAngVelRel(rightShoulderRotStart, rightShoulderRotEnd,
                                                       keyFrameDuration, bullet_client)

        rightElbowRotStart = [frameData[29]]
        rightElbowRotEnd = [frameDataNext[29]]
        self._rightElbowRot = [
            rightElbowRotStart[0] + frameFraction * (rightElbowRotEnd[0] - rightElbowRotStart[0])
        ]
        self._rightElbowVel = [(rightElbowRotEnd[0] - rightElbowRotStart[0]) / keyFrameDuration]

        rightWristRotStart = [frameData[30]]
        rightWristRotEnd = [frameDataNext[30]]
        self._rightWristRot = [
            rightWristRotStart[0] + frameFraction * (rightWristRotEnd[0] - rightWristRotStart[0])
        ]
        self._rightWristVel = [(rightWristRotEnd[0] - rightWristRotStart[0]) / keyFrameDuration]

        rightThumbProxStart = [frameData[31]]
        rightThumbProxEnd = [frameDataNext[31]]
        self._rightThumbProx = [
            rightThumbProxStart[0] + frameFraction * (rightThumbProxEnd[0] - rightThumbProxStart[0])
        ]
        self._rightThumbProxVel = [(rightThumbProxEnd[0] - rightThumbProxStart[0]) / keyFrameDuration]

        rightThumbInterStart = [frameData[32]]
        rightThumbInterEnd = [frameDataNext[32]]
        self._rightThumbInter = [
            rightThumbInterStart[0] + frameFraction * (rightThumbInterEnd[0] - rightThumbInterStart[0])
        ]
        self._rightThumbInterVel = [(rightThumbInterEnd[0] - rightThumbInterStart[0]) / keyFrameDuration]

        rightThumbDistStart = [frameData[33]]
        rightThumbDistEnd = [frameDataNext[33]]
        self._rightThumbDist = [
            rightThumbDistStart[0] + frameFraction * (rightThumbDistEnd[0] - rightThumbDistStart[0])
        ]
        self._rightThumbDistVel = [(rightThumbDistEnd[0] - rightThumbDistStart[0]) / keyFrameDuration]

        rightIndexProxStart = [frameData[34]]
        rightIndexProxEnd = [frameDataNext[34]]
        self._rightIndexProx = [
            rightIndexProxStart[0] + frameFraction * (rightIndexProxEnd[0] - rightIndexProxStart[0])
        ]
        self._rightIndexProxVel = [(rightIndexProxEnd[0] - rightIndexProxStart[0]) / keyFrameDuration]

        rightIndexInterStart = [frameData[35]]
        rightIndexInterEnd = [frameDataNext[35]]
        self._rightIndexInter = [
            rightIndexInterStart[0] + frameFraction * (rightIndexInterEnd[0] - rightIndexInterStart[0])
        ]
        self._rightIndexInterVel = [(rightIndexInterEnd[0] - rightIndexInterStart[0]) / keyFrameDuration]

        rightIndexDistStart = [frameData[36]]
        rightIndexDistEnd = [frameDataNext[36]]
        self._rightIndexDist = [
            rightIndexDistStart[0] + frameFraction * (rightIndexDistEnd[0] - rightIndexDistStart[0])
        ]
        self._rightIndexDistVel = [(rightIndexDistEnd[0] - rightIndexDistStart[0]) / keyFrameDuration]

        rightMiddleProxStart = [frameData[37]]
        rightMiddleProxEnd = [frameDataNext[37]]
        self._rightMiddleProx = [
            rightMiddleProxStart[0] + frameFraction * (rightMiddleProxEnd[0] - rightMiddleProxStart[0])
        ]
        self._rightMiddleProxVel = [(rightMiddleProxEnd[0] - rightMiddleProxStart[0]) / keyFrameDuration]

        rightMiddleInterStart = [frameData[38]]
        rightMiddleInterEnd = [frameDataNext[38]]
        self._rightMiddleInter = [
            rightMiddleInterStart[0] + frameFraction * (rightMiddleInterEnd[0] - rightMiddleInterStart[0])
        ]
        self._rightMiddleInterVel = [(rightMiddleInterEnd[0] - rightMiddleInterStart[0]) / keyFrameDuration]

        rightMiddleDistStart = [frameData[39]]
        rightMiddleDistEnd = [frameDataNext[39]]
        self._rightMiddleDist = [
            rightMiddleDistStart[0] + frameFraction * (rightMiddleDistEnd[0] - rightMiddleDistStart[0])
        ]
        self._rightMiddleDistVel = [(rightMiddleDistEnd[0] - rightMiddleDistStart[0]) / keyFrameDuration]

        rightRingProxStart = [frameData[40]]
        rightRingProxEnd = [frameDataNext[40]]
        self._rightRingProx = [
            rightRingProxStart[0] + frameFraction * (rightRingProxEnd[0] - rightRingProxStart[0])
        ]
        self._rightRingProxVel = [(rightRingProxEnd[0] - rightRingProxStart[0]) / keyFrameDuration]

        rightRingInterStart = [frameData[41]]
        rightRingInterEnd = [frameDataNext[41]]
        self._rightRingInter = [
            rightRingInterStart[0] + frameFraction * (rightRingInterEnd[0] - rightRingInterStart[0])
        ]
        self._rightRingInterVel = [(rightRingInterEnd[0] - rightRingInterStart[0]) / keyFrameDuration]

        rightRingDistStart = [frameData[42]]
        rightRingDistEnd = [frameDataNext[42]]
        self._rightRingDist = [
            rightRingDistStart[0] + frameFraction * (rightRingDistEnd[0] - rightRingDistStart[0])
        ]
        self._rightRingDistVel = [(rightRingDistEnd[0] - rightRingDistStart[0]) / keyFrameDuration]

        rightPinkieProxStart = [frameData[43]]
        rightPinkieProxEnd = [frameDataNext[43]]
        self._rightPinkieProx = [
            rightPinkieProxStart[0] + frameFraction * (rightPinkieProxEnd[0] - rightPinkieProxStart[0])
        ]
        self._rightPinkieProxVel = [(rightPinkieProxEnd[0] - rightPinkieProxStart[0]) / keyFrameDuration]

        rightPinkieInterStart = [frameData[44]]
        rightPinkieInterEnd = [frameDataNext[44]]
        self._rightPinkieInter = [
            rightPinkieInterStart[0] + frameFraction * (rightPinkieInterEnd[0] - rightPinkieInterStart[0])
        ]
        self._rightPinkieInterVel = [(rightPinkieInterEnd[0] - rightPinkieInterStart[0]) / keyFrameDuration]

        rightPinkieDistStart = [frameData[45]]
        rightPinkieDistEnd = [frameDataNext[45]]
        self._rightPinkieDist = [
            rightPinkieDistStart[0] + frameFraction * (rightPinkieDistEnd[0] - rightPinkieDistStart[0])
        ]
        self._rightPinkieDistVel = [(rightPinkieDistEnd[0] - rightPinkieDistStart[0]) / keyFrameDuration]

        leftHipRotStart = [frameData[47], frameData[48], frameData[49], frameData[46]]
        leftHipRotEnd = [frameDataNext[47], frameDataNext[48], frameDataNext[49], frameDataNext[46]]
        self._leftHipRot = bullet_client.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd,
                                                            frameFraction)
        self._leftHipVel = self.ComputeAngVelRel(leftHipRotStart, leftHipRotEnd, keyFrameDuration,
                                                 bullet_client)

        leftKneeRotStart = [frameData[50]]
        leftKneeRotEnd = [frameDataNext[50]]
        self._leftKneeRot = [
            leftKneeRotStart[0] + frameFraction * (leftKneeRotEnd[0] - leftKneeRotStart[0])
        ]
        self._leftKneeVel = [(leftKneeRotEnd[0] - leftKneeRotStart[0]) / keyFrameDuration]

        leftAnkleRotStart = [frameData[52], frameData[53], frameData[54], frameData[51]]
        leftAnkleRotEnd = [frameDataNext[52], frameDataNext[53], frameDataNext[54], frameDataNext[51]]
        self._leftAnkleRot = bullet_client.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd,
                                                              frameFraction)
        self._leftAnkleVel = self.ComputeAngVelRel(leftAnkleRotStart, leftAnkleRotEnd,
                                                   keyFrameDuration, bullet_client)

        leftShoulderRotStart = [frameData[56], frameData[57], frameData[58], frameData[55]]
        leftShoulderRotEnd = [
            frameDataNext[56], frameDataNext[57], frameDataNext[58], frameDataNext[55]
        ]
        self._leftShoulderRot = bullet_client.getQuaternionSlerp(leftShoulderRotStart,
                                                                 leftShoulderRotEnd, frameFraction)
        self._leftShoulderVel = self.ComputeAngVelRel(leftShoulderRotStart, leftShoulderRotEnd,
                                                      keyFrameDuration, bullet_client)

        leftElbowRotStart = [frameData[59]]
        leftElbowRotEnd = [frameDataNext[59]]
        self._leftElbowRot = [
            leftElbowRotStart[0] + frameFraction * (leftElbowRotEnd[0] - leftElbowRotStart[0])
        ]
        self._leftElbowVel = [(leftElbowRotEnd[0] - leftElbowRotStart[0]) / keyFrameDuration]

        leftWristRotStart = [frameData[60]]
        leftWristRotEnd = [frameDataNext[60]]
        self._leftWristRot = [
            leftWristRotStart[0] + frameFraction * (leftWristRotEnd[0] - leftWristRotStart[0])
        ]
        self._leftWristVel = [(leftWristRotEnd[0] - leftWristRotStart[0]) / keyFrameDuration]

        leftThumbProxStart = [frameData[61]]
        leftThumbProxEnd = [frameDataNext[61]]
        self._leftThumbProx = [
            leftThumbProxStart[0] + frameFraction * (leftThumbProxEnd[0] - leftThumbProxStart[0])
        ]
        self._leftThumbProxVel = [(leftThumbProxEnd[0] - leftThumbProxStart[0]) / keyFrameDuration]

        leftThumbInterStart = [frameData[62]]
        leftThumbInterEnd = [frameDataNext[62]]
        self._leftThumbInter = [
            leftThumbInterStart[0] + frameFraction * (leftThumbInterEnd[0] - leftThumbInterStart[0])
        ]
        self._leftThumbInterVel = [(leftThumbInterEnd[0] - leftThumbInterStart[0]) / keyFrameDuration]

        leftThumbDistStart = [frameData[63]]
        leftThumbDistEnd = [frameDataNext[63]]
        self._leftThumbDist = [
            leftThumbDistStart[0] + frameFraction * (leftThumbDistEnd[0] - leftThumbDistStart[0])
        ]
        self._leftThumbDistVel = [(leftThumbDistEnd[0] - leftThumbDistStart[0]) / keyFrameDuration]

        leftIndexProxStart = [frameData[64]]
        leftIndexProxEnd = [frameDataNext[64]]
        self._leftIndexProx = [
            leftIndexProxStart[0] + frameFraction * (leftIndexProxEnd[0] - leftIndexProxStart[0])
        ]
        self._leftIndexProxVel = [(leftIndexProxEnd[0] - leftIndexProxStart[0]) / keyFrameDuration]

        leftIndexInterStart = [frameData[65]]
        leftIndexInterEnd = [frameDataNext[65]]
        self._leftIndexInter = [
            leftIndexInterStart[0] + frameFraction * (leftIndexInterEnd[0] - leftIndexInterStart[0])
        ]
        self._leftIndexInterVel = [(leftIndexInterEnd[0] - leftIndexInterStart[0]) / keyFrameDuration]

        leftIndexDistStart = [frameData[66]]
        leftIndexDistEnd = [frameDataNext[66]]
        self._leftIndexDist = [
            leftIndexDistStart[0] + frameFraction * (leftIndexDistEnd[0] - leftIndexDistStart[0])
        ]
        self._leftIndexDistVel = [(leftIndexDistEnd[0] - leftIndexDistStart[0]) / keyFrameDuration]

        leftMiddleProxStart = [frameData[67]]
        leftMiddleProxEnd = [frameDataNext[67]]
        self._leftMiddleProx = [
            leftMiddleProxStart[0] + frameFraction * (leftMiddleProxEnd[0] - leftMiddleProxStart[0])
        ]
        self._leftMiddleProxVel = [(leftMiddleProxEnd[0] - leftMiddleProxStart[0]) / keyFrameDuration]

        leftMiddleInterStart = [frameData[68]]
        leftMiddleInterEnd = [frameDataNext[68]]
        self._leftMiddleInter = [
            leftMiddleInterStart[0] + frameFraction * (leftMiddleInterEnd[0] - leftMiddleInterStart[0])
        ]
        self._leftMiddleInterVel = [(leftMiddleInterEnd[0] - leftMiddleInterStart[0]) / keyFrameDuration]

        leftMiddleDistStart = [frameData[69]]
        leftMiddleDistEnd = [frameDataNext[69]]
        self._leftMiddleDist = [
            leftMiddleDistStart[0] + frameFraction * (leftMiddleDistEnd[0] - leftMiddleDistStart[0])
        ]
        self._leftMiddleDistVel = [(leftMiddleDistEnd[0] - leftMiddleDistStart[0]) / keyFrameDuration]

        leftRingProxStart = [frameData[70]]
        leftRingProxEnd = [frameDataNext[70]]
        self._leftRingProx = [
            leftRingProxStart[0] + frameFraction * (leftRingProxEnd[0] - leftRingProxStart[0])
        ]
        self._leftRingProxVel = [(leftRingProxEnd[0] - leftRingProxStart[0]) / keyFrameDuration]

        leftRingInterStart = [frameData[71]]
        leftRingInterEnd = [frameDataNext[71]]
        self._leftRingInter = [
            leftRingInterStart[0] + frameFraction * (leftRingInterEnd[0] - leftRingInterStart[0])
        ]
        self._leftRingInterVel = [(leftRingInterEnd[0] - leftRingInterStart[0]) / keyFrameDuration]

        leftRingDistStart = [frameData[72]]
        leftRingDistEnd = [frameDataNext[72]]
        self._leftRingDist = [
            leftRingDistStart[0] + frameFraction * (leftRingDistEnd[0] - leftRingDistStart[0])
        ]
        self._leftRingDistVel = [(leftRingDistEnd[0] - leftRingDistStart[0]) / keyFrameDuration]

        leftPinkieProxStart = [frameData[73]]
        leftPinkieProxEnd = [frameDataNext[73]]
        self._leftPinkieProx = [
            leftPinkieProxStart[0] + frameFraction * (leftPinkieProxEnd[0] - leftPinkieProxStart[0])
        ]
        self._leftPinkieProxVel = [(leftPinkieProxEnd[0] - leftPinkieProxStart[0]) / keyFrameDuration]

        leftPinkieInterStart = [frameData[74]]
        leftPinkieInterEnd = [frameDataNext[74]]
        self._leftPinkieInter = [
            leftPinkieInterStart[0] + frameFraction * (leftPinkieInterEnd[0] - leftPinkieInterStart[0])
        ]
        self._leftPinkieInterVel = [(leftPinkieInterEnd[0] - leftPinkieInterStart[0]) / keyFrameDuration]

        leftPinkieDistStart = [frameData[75]]
        leftPinkieDistEnd = [frameDataNext[75]]
        self._leftPinkieDist = [
            leftPinkieDistStart[0] + frameFraction * (leftPinkieDistEnd[0] - leftPinkieDistStart[0])
        ]
        self._leftPinkieDistVel = [(leftPinkieDistEnd[0] - leftPinkieDistStart[0]) / keyFrameDuration]

        pose = self.GetPose()
        return pose

    def ConvertFromAction(self, pybullet_client, action):
        # turn action into pose

        self.Reset()  # ?? needed?
        index = 0
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        self._chestRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        self._neckRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

        # angle = action[index]
        # axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        # self._rightHipRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
        #
        # angle = action[index]
        index += 1
        # self._rightKneeRot = [angle]
        #
        # angle = action[index]
        # axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        # self._rightAnkleRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
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

        angle = action[index]
        index += 1
        self._rightWristRot = [angle]

        angle = action[index]
        index += 1
        self._rightThumbProx = [angle]

        angle = action[index]
        index += 1
        self._rightThumbInter = [angle]

        angle = action[index]
        index += 1
        self._rightThumbDist = [angle]

        angle = action[index]
        index += 1
        self._rightIndexProx = [angle]

        angle = action[index]
        index += 1
        self._rightIndexInter = [angle]

        angle = action[index]
        index += 1
        self._rightIndexDist = [angle]

        angle = action[index]
        index += 1
        self._rightMiddleProx = [angle]

        angle = action[index]
        index += 1
        self._rightMiddleInter = [angle]

        angle = action[index]
        index += 1
        self._rightMiddleDist = [angle]

        angle = action[index]
        index += 1
        self._rightRingProx = [angle]

        angle = action[index]
        index += 1
        self._rightRingInter = [angle]

        angle = action[index]
        index += 1
        self._rightRingDist = [angle]

        angle = action[index]
        index += 1
        self._rightPinkieProx = [angle]

        angle = action[index]
        index += 1
        self._rightPinkieInter = [angle]

        angle = action[index]
        index += 1
        self._rightPinkieDist = [angle]

        # angle = action[index]
        # axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        # self._leftHipRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
        #
        # angle = action[index]
        index += 1
        # self._leftKneeRot = [angle]
        #
        # angle = action[index]
        # axis = [action[index + 1], action[index + 2], action[index + 3]]
        index += 4
        # self._leftAnkleRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
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

        angle = action[index]
        index += 1
        self._leftWristRot = [angle]

        angle = action[index]
        index += 1
        self._leftThumbProx = [angle]

        angle = action[index]
        index += 1
        self._leftThumbInter = [angle]

        angle = action[index]
        index += 1
        self._leftThumbDist = [angle]

        angle = action[index]
        index += 1
        self._leftIndexProx = [angle]

        angle = action[index]
        index += 1
        self._leftIndexInter = [angle]

        angle = action[index]
        index += 1
        self._leftIndexDist = [angle]

        angle = action[index]
        index += 1
        self._leftMiddleProx = [angle]

        angle = action[index]
        index += 1
        self._leftMiddleInter = [angle]

        angle = action[index]
        index += 1
        self._leftMiddleDist = [angle]

        angle = action[index]
        index += 1
        self._leftRingProx = [angle]

        angle = action[index]
        index += 1
        self._leftRingInter = [angle]

        angle = action[index]
        index += 1
        self._leftRingDist = [angle]

        angle = action[index]
        index += 1
        self._leftPinkieProx = [angle]

        angle = action[index]
        index += 1
        self._leftPinkieInter = [angle]

        angle = action[index]
        index += 1
        self._leftPinkieDist = [angle]

        pose = self.GetPose()
        return pose
