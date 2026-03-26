# HUMAN PLAY
from snake_game import SnakeEnv

env = SnakeEnv(render_mode='human', width=11, height=11)
env.reset()
env.play()
env.close()