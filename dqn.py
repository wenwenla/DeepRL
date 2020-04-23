import os
import gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model.qnet import DQNQNet
from deeprl_util.args import DQNArgs
from deeprl_util.buffer import ReplayBuffer


class Transformer:

    EMPTY = np.zeros((210, 160), dtype=np.float32)

    def __init__(self):
        self._buffer = [Transformer.EMPTY, Transformer.EMPTY, Transformer.EMPTY, Transformer.EMPTY]
        self.shape = (4, 210, 160)

    def clear(self):
        self._buffer = self._buffer[:4]

    def add(self, state):
        state = np.dot(state[..., :3], [0.299, 0.587, 0.114])
        state /= 255.
        self._buffer.append(state)

    def get(self):
        return np.array([self._buffer[i] for i in [-4, -3, -2, -1]], dtype=np.float32)


class DQNAgent:

    def __init__(self, env, args):
        self.env = env
        self.transformer = Transformer()
        self.state_shape = self.transformer.shape
        self.action_cnt = env.action_space.n
        self.qnet = DQNQNet(self.state_shape, self.action_cnt)
        self.target_qnet = DQNQNet(self.state_shape, self.action_cnt)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optim = optim.Adam(self.qnet.parameters(), lr=args.critic_lr)
        self.loss_fn = torch.nn.MSELoss()
        self.args = args
        self.replay = ReplayBuffer(args.exp_cap, self.state_shape, 1)
        self.sw = SummaryWriter(self.args.log_dir)
        self.steps = 0

    def choose_action(self, state):
        state = torch.Tensor([state]).float()
        with torch.no_grad():
            action = self.qnet(state).numpy()
        return np.argmax(action)

    def choose_action_with_exploration(self, state):
        if np.random.uniform() < self.args.epsilon:
            return self.env.action_space.sample()
        return self.choose_action(state)

    def update(self):
        s, a, r, s_, d = self.replay.sample(self.args.batch)
        indices = [_ for _ in range(s.shape[0])]
        with torch.no_grad():
            target = self.qnet(torch.Tensor(s))
            upd = self.args.gamma * self.target_qnet(torch.Tensor(s_))
            upd[d] = 0
            upd = torch.Tensor(r) + torch.max(upd, axis=1)[0]
            target[indices, a] = upd
        self.optim.zero_grad()
        q = self.qnet(torch.Tensor(s))
        loss = self.loss_fn(q, target)
        loss.backward()
        self.optim.step()
        self.soft_copy_parm()
        if self.steps % self.args.log_interval == 0:
            self.sw.add_scalar('loss/qloss', loss.item())

    def soft_copy_parm(self):
        with torch.no_grad():
            for t, s in zip(self.target_qnet.parameters(), self.qnet.parameters()):
                t.copy_(0.95 * t.data + 0.05 * s.data)

    def train_one_episode(self):
        state = self.env.reset()
        self.transformer.clear()
        self.transformer.add(state)
        state = self.transformer.get()
        done = False
        total = 0
        while not done:
            action = self.choose_action_with_exploration(state)
            state_, reward, done, _ = self.env.step(action)
            total += reward
            self.transformer.add(state_)
            state_ = self.transformer.get()
            self.replay.add(state, action, reward, state_, done)
            self.update()
            state = state_
            self.steps += 1
            self.sw.add_scalar('reward/train', total, self.steps)
        return total

    def test_one_episode(self):
        state = self.env.reset()
        self.transformer.clear()
        self.transformer.add(state)
        state = self.transformer.get()
        done = False
        total = 0
        while not done:
            action = self.choose_action(state)
            state_, reward, done, _ = self.env.step(action)
            total += reward
            self.transformer.add(state_)
            state_ = self.transformer.get()
            state = state_
        return total

    def test_model(self, cnt=10):
        r = [self.test_one_episode() for _ in range(cnt)]
        r_mean = np.mean(r)
        self.sw.add_scalar('reward/test', r_mean, self.steps)
        return r_mean

    def save(self, path):
        path = os.path.join(path, 'best.pkl')
        state_dict = self.qnet.state_dict()
        torch.save(state_dict, path)


def train_dqn():
    args = DQNArgs()
    env = gym.make(args.env_name)
    agent = DQNAgent(env, args)
    pre_best = -1e9
    for ep in range(args.max_ep):
        agent.train_one_episode()
        if ep % args.test_interval == 0:
            r = agent.test_model()
            if r > pre_best:
                pre_best = r
                agent.save(args.save_dir)


if __name__ == '__main__':
    train_dqn()
