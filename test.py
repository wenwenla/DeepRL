import os
import gym
from ddpg import DDPGAgent
from deeprl_util.args import Args


def test_ddpg():
    args = Args()
    env = gym.make(args.env_name)
    agent = DDPGAgent(env, args)
    agent.load(os.path.join(args.save_path, 'best.pkl'))
    for _ in range(10):
        agent.test_one_episode(True)


if __name__ == '__main__':
    test_ddpg()