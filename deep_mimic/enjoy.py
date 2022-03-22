import time
import torch.nn as nn
from stable_baselines3 import PPO
import deep_mimic
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np


log_dir = "output/2022-03-21 11:54:15.830073/"
env_name = 'HandDeepMimicSignerBulletEnv-v1'

env = make_vec_env(env_name)
env = VecNormalize.load(log_dir+"vecnormalize.pkl", env)

model = PPO.load(log_dir+"best_model", env=env)

env.render(mode='human')

obs = env.reset()
dones = [False]
tot_rew = 0
actions = []
while not all(dones):
    action, _states = model.predict(obs, deterministic=True)
    actions.append(env.env_method("unscale_action", action)[0])
    obs, rewards, dones, info = env.step(action)
    time.sleep(1./240.)
    tot_rew += rewards

print(tot_rew)

actions = np.array(actions)

for i in range(15):
    min = np.min(actions[:, 0, i])
    max = np.max(actions[:, 0, i])
    print(i, min, max)

