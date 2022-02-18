import time
import torch.nn as nn
from stable_baselines3 import PPO
import deep_mimic
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

log_dir = "gym_env/output/"
env_name = 'HandDeepMimicSignerBulletEnv-v1'

env = make_vec_env(env_name)
env = VecNormalize.load(log_dir+"vecnormalize.pkl", env)

model = PPO.load(log_dir+"best_model", env=env)

env.render(mode='human')

obs = env.reset()
dones = [False]
tot_rew = 0
while not all(dones):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    time.sleep(1./30.)
    tot_rew += rewards

print(tot_rew)
