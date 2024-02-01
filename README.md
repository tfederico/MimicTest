# Debug code for mimicry

To train using PPO
`python -m deep_mimic.train_ppo`

To train using SAC
`python -m deep_mimic.train_sac`

To change from whole body to hand, change the string `env_name` in `train_*.py`.

`HandDeepMimicSignerBulletEnv-v1` is for the hand, `WholeDeepMimicSignerBulletEnv-v1` is for the whole body
