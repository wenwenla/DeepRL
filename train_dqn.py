import torch
import torch.nn as nn
import gym
import numpy as np
from dqn import DQNAgent
from ddqn import DDQNAgent
from deeprl_util.args import DQNArgs, DDQNArgs
from deeprl_util.preprocessing import SimpleNormalizer


class QNet(nn.Module):

    def __init__(self, state_shape, action_cnt):
        super().__init__()
        self.fc0 = nn.Linear(state_shape[0], 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, action_cnt)

    def forward(self, x):
        x = torch.tanh(self.fc0(x))
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


def train_dqn():
    args = DQNArgs()
    env = gym.make(args.env_name)
    agent = DQNAgent(env, QNet, SimpleNormalizer, args)
    pre_best = -1e9
    for ep in range(args.max_ep):
        agent.train_one_episode()
        if ep % args.test_interval == 0:
            r = agent.test_model()
            if r > pre_best:
                pre_best = r
                agent.save(args.save_dir)


def test_dqn():
    args = DQNArgs()
    env = gym.make(args.env_name)
    agent = DQNAgent(env, QNet, SimpleNormalizer, args)
    agent.load(args.save_dir)
    for _ in range(10):
        agent.test_one_episode(True)


def train_ddqn():
    args = DDQNArgs()
    env = gym.make(args.env_name)
    agent = DDQNAgent(env, QNet, SimpleNormalizer, args)
    pre_best = -1e9
    for ep in range(args.max_ep):
        agent.train_one_episode()
        if ep % args.test_interval == 0:
            r = agent.test_model()
            if r > pre_best:
                pre_best = r
                agent.save(args.save_dir)


def test_ddqn():
    args = DDQNArgs()
    env = gym.make(args.env_name)
    agent = DDQNAgent(env, QNet, SimpleNormalizer, args)
    agent.load(args.save_dir)
    mean_reward = [agent.test_one_episode(True) for _ in range(100)]
    print(np.mean(mean_reward))


if __name__ == '__main__':
    # train_dqn()
    # test_dqn()
    # train_ddqn()
    # train_dqn()
    test_ddqn()
