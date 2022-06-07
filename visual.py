import pybullet as p
import pybullet_data


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setRealTimeSimulation(True)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF

p.configureDebugVisualizer(p.COV_ENABLE_GUI,0, rgbBackground=[255, 255, 255])


id = p.loadURDF("/home/federico/Git/ASLMimic/data/humanoid/whole_upper_humanoid.urdf", basePosition=[0, 0, -0.5], baseOrientation=[ 0.7071068, 0, 0, 0.7071068 ], useFixedBase=True)

input("Press Enter to continue...")
p.disconnect()