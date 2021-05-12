import gym
from gym.envs.registration import registry, make, spec

def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)


# ------------bullet-------------

register(
    id='HumanoidDeepMimicUpperSignerBulletEnv-v1',
    entry_point='deep_mimic.gym_env:HumanoidDeepMimicUpperSignerBulletEnv',
    max_episode_steps=2000,
    reward_threshold=2000.0,
)

register(
    id='HumanoidDeepMimicSignerBulletEnv-v1',
    entry_point='deep_mimic.gym_env:HumanoidDeepMimicSignerBulletEnv',
    max_episode_steps=2000,
    reward_threshold=2000.0,
)

register(
    id='HumanoidDeepMimicDancerBulletEnv-v1',
    entry_point='deep_mimic.gym_env:HumanoidDeepMimicDancerBulletEnv',
    max_episode_steps=2000,
    reward_threshold=2000.0,
)

register(
    id='HumanoidDeepMimicWalkerBulletEnv-v1',
    entry_point='deep_mimic.gym_env:HumanoidDeepMimicWalkerBulletEnv',
    max_episode_steps=2000,
    reward_threshold=2000.0,
)

register(
    id='HumanoidDeepMimicBackflipBulletEnv-v1',
    entry_point='deep_mimic.gym_env:HumanoidDeepMimicBackflipBulletEnv',
    max_episode_steps=2000,
    reward_threshold=2000.0,
)

register(
    id='HumanoidDeepMimicWalkBulletEnv-v1',
    entry_point='deep_mimic.gym_env:HumanoidDeepMimicWalkBulletEnv',
    max_episode_steps=2000,
    reward_threshold=2000.0,
)