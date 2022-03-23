import os
import gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from deep_mimic.gym_env.custom_callbacks import ProgressBarManager, TensorboardCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import deep_mimic
import wandb
import argparse
from wandb.integration.sb3 import WandbCallback
from datetime import datetime
import multiprocessing



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    time = datetime.now()
    run = wandb.init(
        project="test",
        config=args,
        sync_tensorboard=True,
        monitor_gym=False
    )
    args = wandb.config

    # Create log dir

    log_dir = "output/"+str(time)
    os.makedirs(log_dir, exist_ok=True)

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[1024, 512], vf=[1024, 512])],
        log_std_init=args.log_std_init,
        ortho_init=args.ortho_init,
        optimizer_kwargs=dict(weight_decay=args.weight_decay)
    )
    model_args = dict(
        norm_reward=False,
        norm_obs=True,
        learning_rate=args.learning_rate,
        n_steps=4096,
        batch_size=256,
        n_epochs=3,
        gamma=args.gamma,
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

    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=log_dir)
    tensorboard_callback = TensorboardCallback(verbose=0)
    wandb_callback = WandbCallback()
    # Separate evaluation env
    eval_env = make_vec_env(env_name)
    eval_env = VecNormalize(eval_env, norm_reward=model_args['norm_reward'], norm_obs=model_args['norm_obs'])
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, n_eval_episodes=100,
                                 eval_freq=50000, deterministic=True)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback, wandb_callback, tensorboard_callback])

    n_envs = 100

    #env = make_vec_env(env_name, n_envs=n_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
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
    env.save(log_dir+"/vecnormalize.pkl")
    n_steps = int(10e7)
    with ProgressBarManager(n_steps) as prog_callback: # tqdm progress bar closes correctly
        model.learn(n_steps, callback=[prog_callback, callback])

    env.save(log_dir+"/vecnormalize.pkl")
    env.close()
    run.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_std_init', type=int, default=-3)
    parser.add_argument('--ortho_init', type=str2bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=3.0e-6)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=1.0e-5)

    args = parser.parse_args()

    main(args)