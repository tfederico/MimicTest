from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pybullet_envs
import torch.nn as nn
import time

env_name = 'HandDeepMimicSignerBulletEnv-v1'

env = make_vec_env(env_name)
env = VecNormalize.load("output/vecnormalize.pkl", env)

model = PPO.load("output/best_model", env=env)

avg, std = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

print("{} +/- {}".format(avg, std))