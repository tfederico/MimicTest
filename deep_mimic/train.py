import os
import gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from deep_mimic.gym_env.custom_callbacks import ProgressBarManager, TensorboardCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import deep_mimic

# Create log dir
log_dir = "output/"
os.makedirs(log_dir, exist_ok=True)

policy_kwargs = dict(
    activation_fn=nn.ReLU,
    net_arch=[dict(pi=[1024, 512], vf=[1024, 512])],
    log_std_init=-3,
    ortho_init=True,
    optimizer_kwargs=dict(weight_decay=1.0e-5)
)
model_args = dict(
    norm_reward=False,
    norm_obs=True,
    learning_rate=3.0e-6,
    n_steps=4096,
    batch_size=256,
    n_epochs=3,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0,
    vf_coef=5.,
    max_grad_norm=100.,
    target_kl=0.05,
    tensorboard_log=log_dir,
    policy_kwargs=policy_kwargs,
    seed=8
)

env_name = 'HandDeepMimicSignerBulletEnv-v1'

checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path=log_dir)
tensorboard_callback = TensorboardCallback(verbose=0)
# Separate evaluation env
eval_env = make_vec_env(env_name)
eval_env = VecNormalize(eval_env, norm_reward=model_args['norm_reward'], norm_obs=model_args['norm_obs'])
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, n_eval_episodes=10,
                             eval_freq=10000, deterministic=True)
# Create the callback list
callback = CallbackList([checkpoint_callback, tensorboard_callback, eval_callback])

n_envs = 8
env = DummyVecEnv([lambda: Monitor(gym.make(env_name), log_dir) for _ in range(n_envs)])
env = VecNormalize(env, norm_reward=model_args['norm_reward'], norm_obs=model_args['norm_obs'])

model = PPO(
    'MlpPolicy',
    env,
    learning_rate=model_args['learning_rate'],
    n_steps=model_args['n_steps'],
    batch_size=model_args['batch_size'],
    n_epochs=model_args['n_epochs'],
    gamma=model_args['gamma'],
    gae_lambda=model_args['gae_lambda'],
    clip_range=model_args['clip_range'],
    ent_coef=model_args['ent_coef'],
    vf_coef=model_args['vf_coef'],
    max_grad_norm=model_args['max_grad_norm'],
    target_kl=model_args['target_kl'],
    tensorboard_log=model_args['tensorboard_log'],
    policy_kwargs=model_args['policy_kwargs'],
    seed=model_args['seed']
)
env.save(log_dir+"vecnormalize.pkl")
n_steps = int(1e6)
with ProgressBarManager(n_steps) as prog_callback: # tqdm progress bar closes correctly
    model.learn(n_steps, callback=[prog_callback, callback])

env.save(log_dir+"vecnormalize.pkl")