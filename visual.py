import pybullet as p
import pybullet_data


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setRealTimeSimulation(True)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf", [0, 0, -1])
id = p.loadURDF("/home/federico/Git/ASLMimic/data/humanoid/hand.urdf", useFixedBase=True)

n = p.getNumJoints(id)
print(p.getJointInfo(id, 1))

max_force = 500

# for i in range(n):
# p.resetJointStateMultiDof(id, 1, [0, 0, 1, 0], [0, 0, 0])
p.resetJointStateMultiDof(id, 2, [1], [0])

# p.setJointMotorControlMultiDof(bodyUniqueId=id, jointIndex=1, controlMode=p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1], targetVelocity=[0, 0, 0], force=[max_force]*3)
# p.setJointMotorControlMultiDof(bodyUniqueId=id, jointIndex=2, controlMode=p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1], targetVelocity=[0, 0, 0], force=[max_force]*3)
# p.setJointMotorControl2(bodyIndex=id, jointIndex=3, controlMode=p.POSITION_CONTROL, targetPosition=3.14, force=max_force)
# p.setJointMotorControl2(bodyIndex=id, jointIndex=4, controlMode=p.POSITION_CONTROL, targetPosition=3.14, force=max_force)
# p.setJointMotorControl2(bodyIndex=id, jointIndex=5, controlMode=p.POSITION_CONTROL, targetPosition=3.14, force=max_force)

for i in range(1000):
    p.stepSimulation(physicsClient)

print(p.getJointInfo(id, 1))

input("Press Enter to continue...")
p.disconnect()