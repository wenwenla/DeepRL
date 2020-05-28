import os
import pickle
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler


class Configuration:

    def __init__(self):
        self.n_iter = 100
        self.batch_size = 5000
        self.gamma = 0.99
        self.lr = 5e-3
        self.env_name = 'CartPole-v0'
        self.rtg = False
        self.dsa = True
        tag = ''
        if self.rtg:
            tag += 'rtg'
        else:
            tag += 'nortg'
        if self.dsa:
            tag += '-dsa'
        else:
            tag += '-nodsa'
        self.log_step = 10
        self.model_path = './result/PG/{}-{}-{}'.format(self.env_name, self.batch_size, tag)
        self.log_path = './logs/PG/{}-{}-{}'.format(self.env_name, self.batch_size, tag)


class Normalizer:

    def __init__(self):
        self._norm = StandardScaler()

    def transform(self, state):
        self._norm.partial_fit([state])
        return self._norm.transform([state])[0]

    def save(self, path):
        fp = os.path.join(path, 'normalizer.pkl')
        with open(fp, 'wb') as f:
            pickle.dump(self._norm, f)

    def load(self, path):
        fp = os.path.join(path, 'normalizer.pkl')
        with open(fp, 'rb') as f:
            self._norm = pickle.load(f)


class PNet(nn.Module):

    def __init__(self, state_shape, action_cnt):
        super().__init__()
        self.fc0 = nn.Linear(state_shape[0], 64)
        self.fc1 = nn.Linear(64, action_cnt)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = self.fc1(x)
        return self.softmax(x)


class PGAgent:

    def __init__(self, args):
        self._args = args
        self._env = gym.make(self._args.env_name)
        self._norm = Normalizer()
        self._pnet = PNet(self._env.observation_space.shape, self._env.action_space.n)
        self._optim = optim.Adam(self._pnet.parameters(), lr=args.lr)
        self._s, self._a, self._r = [], [], []
        self._step = 0
        if not os.path.exists(self._args.model_path):
            os.makedirs(self._args.model_path)
        self._sw = SummaryWriter(self._args.log_path)

    def choose_action(self, state, deterministic):
        state = torch.FloatTensor([state])
        with torch.no_grad():
            probs = self._pnet(state)
            if not deterministic:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().detach().cpu().numpy()[0]
            else:
                action = np.argmax(probs.detach().cpu().numpy()[0])
        return action

    def _calc_return(self, r):
        g = [r[-1]]
        for v in r[-1:0:-1]:
            g.append(v + self._args.gamma * g[-1])
        ret = g[::-1]
        if not self._args.rtg:
            for i in range(1, len(ret)):
                ret[i] = ret[0]
        return ret

    def _update(self, s, a, r):
        s_feed = torch.FloatTensor(s)
        a_feed = torch.LongTensor(a) # NO NEED TO RESHAPE
        r_feed = torch.FloatTensor(r) # NO NEED TO RESHAPE
        if self._args.dsa:
            r_feed -= r_feed.mean()
        # print('S_feed Shape: {}'.format(s_feed.shape))
        # print('A_feed Shape: {}'.format(a_feed.shape))
        # print('R_Feed Shape: {}'.format(r_feed.shape))
        probs = self._pnet(s_feed)
        # print('PROBS: {}'.format(probs.shape))
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(a_feed)
        # print('LOG_PROBS: {}'.format(log_probs.shape))
        loss = -(log_probs * r_feed).mean()
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        if self._step % self._args.log_step == 0:
            self._sw.add_scalar('loss/pnet', loss.detach().cpu().item(), self._step)

    def _train_one_step(self):
        batch = self._args.batch_size
        while len(self._s) < batch:
            s, a, r = self._gen_trajectory(False)
            g = self._calc_return(r)
            self._s.extend(s)
            self._a.extend(a)
            self._r.extend(g)
        self._update(self._s[:batch], self._a[:batch], self._r[:batch])
        self._s = self._s[batch:]
        self._a = self._a[batch:]
        self._r = self._r[batch:]

    def _gen_trajectory(self, deterministic):
        s, a, r = [], [], []
        state = self._env.reset()
        state = self._norm.transform(state)
        done = False
        while not done:
            action = self.choose_action(state, deterministic)
            state_, reward, done, _ = self._env.step(action)
            state_ = self._norm.transform(state_)
            s.append(state)
            a.append(action)
            r.append(reward)
            state = state_
        return s, a, r

    def test_model(self, n=10):
        rewards = [sum(self._gen_trajectory(True)[2]) for _ in range(n)]
        return np.mean(rewards)

    def save(self):
        self._norm.save(self._args.model_path)
        model_fp = os.path.join(self._args.model_path, 'model.pkl')
        torch.save(self._pnet.state_dict(), model_fp)

    def train(self):
        best = -1e9
        for ep in range(0, 1 + self._args.n_iter):
            self._step = ep
            if self._step % self._args.log_step == 0:
                r = self.test_model()
                self._sw.add_scalar('reward/test', r, self._step)
                if r >= best:
                    self.save()
                    best = r
            self._train_one_step()


if __name__ == '__main__':
    configuration = Configuration()
    pg_agent = PGAgent(configuration)
    pg_agent.train()
