import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from deeprl_util.buffer import ReplayBuffer


class DQNAgent:

    def __init__(self, env, qnet_cls, preprocessing_cls, args):
        self.env = env
        self.pre = preprocessing_cls(env)
        self.state_shape = env.observation_space.shape
        self.action_cnt = env.action_space.n
        self.qnet = qnet_cls(self.state_shape, self.action_cnt)
        self.target_qnet = qnet_cls(self.state_shape, self.action_cnt)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optim = optim.Adam(self.qnet.parameters(), lr=args.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.args = args
        self.replay = ReplayBuffer(args.exp_cap, self.state_shape, 1)
        self.sw = SummaryWriter(self.args.log_dir)
        self.steps = 0
        self.episode = 0
        self._now_epsilon = args.max_epsilon
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

    def choose_action(self, state):
        state = torch.Tensor([state]).float()
        with torch.no_grad():
            action = self.qnet(state).numpy()
        return np.argmax(action)

    def choose_action_with_exploration(self, state):
        if np.random.uniform() < self._now_epsilon:
            return self.env.action_space.sample()
        return self.choose_action(state)

    def update(self):
        s, a, r, s_, d = self.replay.sample(self.args.batch)
        with torch.no_grad():
            target = self.qnet(torch.Tensor(s))
            nxt_q = self.target_qnet(torch.Tensor(s_)).max(axis=1)[0]
            upd = self.args.gamma * nxt_q
            upd = torch.Tensor(r) + upd
            for i, v in enumerate(a):
                target[i, v] = r[i] if d[i] else upd[i]
        self.optim.zero_grad()
        q = self.qnet(torch.Tensor(s))
        loss = self.loss_fn(q, target)
        loss.backward()
        self.optim.step()
        self.hard_copy_parm()
        if self.steps % self.args.log_interval == 0:
            self.sw.add_scalar('loss/qloss', loss.item(), self.steps)

    def hard_copy_parm(self):
        if self.steps % self.args.tau == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

    def train_one_episode(self):
        state = self.env.reset()
        self.pre.reset()
        state = self.pre.transform(state)
        done = False
        total = 0
        while not done:
            action = self.choose_action_with_exploration(state)
            state_, reward, done, _ = self.env.step(action)
            total += reward
            state_ = self.pre.transform(state_)
            self.replay.add(state, action, reward, state_, done)
            self.update()
            state = state_
            self.steps += 1
        self.episode += 1
        self._now_epsilon -= self.args.epsilon_decay
        self._now_epsilon = max(self._now_epsilon, self.args.min_epsilon)
        self.sw.add_scalar('reward/train', total, self.episode)
        self._log_avg_q()
        return total

    def test_one_episode(self, viewer=False):
        state = self.env.reset()
        self.pre.reset()
        state = self.pre.transform(state)
        done = False
        total = 0
        while not done:
            action = self.choose_action(state)
            state_, reward, done, _ = self.env.step(action)
            if viewer:
                self.env.render()
                time.sleep(0.1)
            total += reward
            state_ = self.pre.transform(state_)
            state = state_
        return total

    def test_model(self, cnt=10):
        r = [self.test_one_episode() for _ in range(cnt)]
        r_mean = np.mean(r)
        self.sw.add_scalar('reward/test', r_mean, self.episode)
        return r_mean

    def save(self, path):
        self.pre.save(path)
        path = os.path.join(path, 'best.pkl')
        state_dict = self.qnet.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        self.pre.load(path)
        path = os.path.join(path, 'best.pkl')
        state_dict = torch.load(path)
        self.qnet.load_state_dict(state_dict)

    def _log_avg_q(self):
        s, *_ = self.replay.sample(64)
        s_feed = torch.FloatTensor(s)
        with torch.no_grad():
            q = self.qnet(s_feed)
            val = q.mean().item()
        self.sw.add_scalar('avg_q', val, self.episode)