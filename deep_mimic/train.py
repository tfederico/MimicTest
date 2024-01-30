import os
import gym
import torch.nn as nn
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from deep_mimic.gym_env.custom_callbacks import ProgressBarManager, TensorboardCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import deep_mimic
import wandb
import argparse
from wandb.integration.sb3 import WandbCallback
from datetime import datetime


def str2int(v):
    return [int(x) for x in v.split(' ')]


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
        net_arch=[int(x) for x in args.pi_vf.split(" ")],
        log_std_init=args.log_std_init,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=args.weight_decay)
    )
    model_args = dict(
        norm_reward=False,
        norm_obs=True,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        gamma=0.99,
        tensorboard_log=log_dir,
        seed=args.seed,
    )

    env_name = 'WholeDeepMimicSignerBulletEnv-v1'

    checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path=log_dir)
    tensorboard_callback = TensorboardCallback(verbose=0)
    wandb_callback = WandbCallback()
    # Separate evaluation env
    eval_env = make_vec_env(env_name, env_kwargs=dict(renders=False,
                                                      arg_file=f"run_humanoid3d_{args.motion_file}_args.txt"))
    eval_env = VecNormalize(eval_env, norm_reward=model_args['norm_reward'], norm_obs=model_args['norm_obs'])
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir, n_eval_episodes=10,
                                 eval_freq=5000, deterministic=True)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback, wandb_callback, tensorboard_callback])

    n_envs = 10

    # env = make_vec_env(env_name, n_envs=n_envs, vec_env_cls=SubprocVecEnv, monitor_dir=log_dir,
    #                    env_kwargs=dict(renders=False, arg_file=f"run_humanoid3d_{args.motion_file}_args.txt"),
    #                    vec_env_kwargs=dict(start_method='fork'))
    env = DummyVecEnv([lambda: Monitor(gym.make(env_name,
                                                **dict(renders=False,
                                                       arg_file=f"run_humanoid3d_{args.motion_file}_args.txt")),
                                       log_dir) for _ in range(n_envs)])
    env = VecNormalize(env, norm_reward=model_args['norm_reward'], norm_obs=model_args['norm_obs'])

    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=model_args['learning_rate'],
        buffer_size=model_args['buffer_size'],
        learning_starts=model_args['learning_starts'],
        batch_size=model_args['batch_size'],
        tau=model_args['tau'],
        train_freq=model_args['train_freq'],
        gradient_steps=model_args['gradient_steps'],
        gamma=model_args['gamma'],
        ent_coef=model_args['ent_coef'],
        tensorboard_log=model_args['tensorboard_log'],
        policy_kwargs=policy_kwargs,
        seed=model_args['seed']
    )
    env.save(log_dir+"/vecnormalize.pkl")
    n_steps = args.glob_n_steps
    with ProgressBarManager(n_steps) as prog_callback: # tqdm progress bar closes correctly
        model.learn(n_steps, callback=[prog_callback, callback])

    env.save(log_dir+"/vecnormalize.pkl")
    env.close()
    run.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_file', type=str, default="tuning_motion_whole")
    parser.add_argument('--glob_n_steps', type=int, default=2e7)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--learning_starts', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--gradient_steps', type=int, default=1)
    parser.add_argument('--ent_coef', type=str, default="auto_0.1")
    parser.add_argument('--target_entropy', type=str, default="auto")
    parser.add_argument('--log_std_init', type=float, default=-3.0)
    parser.add_argument('--weight_decay', type=float, default=1.0e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pi_vf', type=str, default="1024 512")

    args = parser.parse_args()

    main(args)