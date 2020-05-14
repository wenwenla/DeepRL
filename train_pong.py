import gym
from deeprl_util.preprocessing import DQNTransformer


env = gym.make('PongNoFrameskip-v4')
s = env.reset()

pre = DQNTransformer(env)
pre.reset()
pre.transform(s)