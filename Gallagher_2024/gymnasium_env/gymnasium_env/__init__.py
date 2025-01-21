from gymnasium.envs.registration import register

register(
    id="gymnasium_env/LotkaVolterra-v0",
    entry_point="gymnasium_env.envs:LotkaVolterraEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)
