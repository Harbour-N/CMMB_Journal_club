from gymnasium.envs.registration import register

register(
    id="gymnasium_env/tumor_model-v0",
    entry_point="gymnasium_env.envs:tumor_model",
)
