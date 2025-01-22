
# run_gymnasium_env.py

import gymnasium
import gymnasium_env
env = gymnasium.make('gymnasium_env/tumor_model-v0')

env.reset()
env.render()

