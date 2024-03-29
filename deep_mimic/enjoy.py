import time
import torch.nn as nn
from stable_baselines3 import PPO, SAC
import deep_mimic
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--motion_file', type=str, default="tuning_motion_whole")
args = parser.parse_args()

# # otho init true, lr 3.0e-5
# dirs = {
#     "A": "2022-05-05 08:56:10.350132",
#     "B": "2022-05-05 09:58:10.960467",
#     "C": "2022-05-05 22:52:49.513238",
#     "D": "2022-05-05 23:19:31.016639",
#     "E": "2022-05-05 23:30:26.339840",
#     "F": "2022-05-06 00:07:51.075258"
# }

# # otho init true, lr 3.0e-6
# dirs = {
#     "A": "2022-05-06 09:37:16.144619",
#     "B": "2022-05-06 12:55:03.110993",
#     "C": "2022-05-06 13:22:00.727932",
#     "D": "2022-05-06 14:43:06.459355",
#     "E": "2022-05-06 23:11:58.718173",
#     "F": "2022-05-07 05:27:35.639943"
# }

# # otho init false, lr 3.0e-5
# dirs = {
#     "A": "2022-05-03 18:06:07.285398",
#     "B": "2022-05-03 18:06:07.286085",
#     "C": "2022-05-03 18:06:07.285951",
#     "D": "2022-05-03 18:06:07.258203",
#     "E": "2022-05-04 04:18:05.536432",
#     "F": "2022-05-04 04:35:18.399239"
# }

# # otho init false, lr 3.0e-6
# dirs = {
#     "A": "2022-05-04 17:34:34.313663",
#     "B": "2022-05-04 17:34:34.313271",
#     "C": "2022-05-04 17:34:34.312249",
#     "D": "2022-05-04 17:34:34.312162",
#     "E": "2022-05-05 07:45:15.073633",
#     "F": "2022-05-05 08:52:03.644891"
# }

# # otho init true, lr 3.0e-6, short motions
# dirs = {
#     "A": "2022-05-11 08:21:03.759462",
#     "B": "2022-05-11 08:18:06.622367",
#     "C": "2022-05-10 18:13:14.107628",
#     "D": "2022-05-10 18:13:14.109400",
#     "E": "2022-05-10 18:13:14.109346",
#     "F": "2022-05-10 18:13:14.109208"
# }

# # short motions, default params
# dirs = {
#     "A": "2022-05-11 08:21:03.759462",
#     "B": "2022-05-11 08:18:06.622367",
#     "C": "2022-05-10 18:13:14.107628",
#     "D": "2022-05-10 18:13:14.109400",
#     "E": "2022-05-10 18:13:14.109346",
#     "F": "2022-05-10 18:13:14.109208"
# }

# # final final motions
# dirs = {
#     "A": "2022-06-09 14:13:02.985153",
#     "B": "2022-06-13 10:53:45.209827",
#     "C": "2022-06-15 10:09:37.293896",
#     "D": "2022-06-15 10:09:56.387131",
#     "E": "2022-06-13 19:44:47.155149",
#     "F": "2022-06-14 15:03:46.209549",
#     "tuning_motion": "2022-06-08 05:28:43.743301"
# }

dirs = {
    "tuning_motion_whole": "2024-01-30 17:54:49.043492",
    "00433": "2022-11-02 09:59:47.226551", #"2022-10-24 16:00:23.254204",  # "2022-10-20 14:27:10.251318",
    "52861": "2022-11-02 09:59:45.739750", #"2022-10-24 16:00:43.085142",  # "2022-10-20 14:27:52.137148",
    "69318": "",
    "69402": "",
    "69546": ""
}


log_dir = f"output/{dirs[args.motion_file]}/"
env_name = 'WholeDeepMimicSignerBulletEnv-v1'


env = make_vec_env(env_name, env_kwargs=dict(renders=True, arg_file=f"run_humanoid3d_{args.motion_file}_args.txt", test_mode=True))

env = VecNormalize.load(log_dir+"vecnormalize.pkl", env)

model = SAC.load(log_dir+"best_model", env=env)

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
