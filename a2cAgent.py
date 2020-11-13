from collections import namedtuple

import gym
import torch
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
from models import VNet, PolicyNet


class A2CArgs:

    def __init__(self, p_lr=1e-4, v_lr=1e-3, max_episode=int(1e5), gamma=0.99, entropy_alpha=0.001, log_dir='./logs',
                 length=25):
        self.p_lr = p_lr
        self.v_lr = v_lr
        self.max_episode = max_episode
        self.gamma = gamma
        self.entropy_alpha = entropy_alpha
        self.log_dir = log_dir
        self.length = length


class A2CAgent:

    Transition = namedtuple('Transition', [
        'prev_state', 'action', 'reward', 'next_state', 'done'
    ])

    def __init__(self, env: gym.Env, args: A2CArgs, verbose=False):
        self.state_dim = env.observation_space.shape[0]
        self.action_cnt = env.action_space.n
        self.v_net = VNet(self.state_dim)
        self.p_net = PolicyNet(self.state_dim, self.action_cnt)
        self.v_opt = opt.Adam(self.v_net.parameters(), lr=args.v_lr)
        self.p_opt = opt.Adam(self.p_net.parameters(), lr=args.p_lr)

        self._env = env
        self._args = args

        self._verbose = verbose
        if self._verbose:
            self._sw = SummaryWriter(args.log_dir)

        # log info
        self._now_ep = 0
        self._now_step = 0

    def _update_v(self, states, g):
        s = torch.tensor(states).view(-1, self.state_dim).float()
        g = torch.tensor(g).view(-1, 1).float()
        v = self.v_net(s)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(v, g)
        self.v_opt.zero_grad()
        loss.backward()
        self.v_opt.step()

        if self._verbose:
            self._sw.add_scalar('loss/v_loss', loss.detach().item(), self._now_step)

    def _update_policy(self, states, actions, next_states):
        s = torch.tensor(states).view(-1, self.state_dim).float()
        s_ = torch.tensor(next_states).view(-1, self.state_dim).float()
        with torch.no_grad():
            label = self._args.gamma * self.v_net(s_) - self.v_net(s)
            label = label.view(-1)
        a = torch.tensor(actions).view(-1).float()
        policy = self.p_net(s)
        dist = torch.distributions.Categorical(policy)
        log_prob = dist.log_prob(a)

        loss = -(log_prob * label).mean() - self._args.entropy_alpha * dist.entropy().mean()  # 最大化累积奖赏的同时，最大化策略的熵
        self.p_opt.zero_grad()
        loss.backward()
        self.p_opt.step()

        if self._verbose:
            self._sw.add_scalar('loss/p_loss', loss.detach().item(), self._now_step)
            self._sw.add_scalar('entropy/p_entropy', dist.entropy().mean().detach().item(), self._now_step)

    def _update(self, transitions):
        s, a, s_, g = [], [], [], []

        successive_rewards = 0
        if not transitions[-1].done:
            with torch.no_grad():
                # compute V(s')
                successive_rewards = self.v_net(torch.tensor([transitions[-1].next_state]).float())[0].item()

        for t in reversed(transitions):
            if t.done:
                successive_rewards = 0
            s.append(t.prev_state)
            g.append(t.reward + self._args.gamma * successive_rewards)
            a.append(t.action)
            s_.append(t.next_state)

        self._now_step += 1
        # update V
        self._update_v(s, g)
        # update Pi
        self._update_policy(s, a, s_)

    def choose_action(self, state):
        feed_in_state = torch.tensor([state]).float()
        with torch.no_grad():
            action_prob = self.p_net(feed_in_state)[0]
            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()
        return action.item()

    def train(self):
        transitions = []

        for ep in range(self._args.max_episode):
            self._now_ep = ep
            total_r = 0
            state = self._env.reset()

            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self._env.step(action)
                total_r += reward

                transitions.append(A2CAgent.Transition(state, action, reward, next_state, done))

                if len(transitions) == self._args.length:
                    self._update(transitions)
                    transitions.clear()
                state = next_state

            if self._verbose:
                self._sw.add_scalar('return', total_r, ep)
            print(f'EP: {ep}, TOTAL: {total_r}')


def main():
    env = gym.make('CartPole-v0')
    args = A2CArgs(length=30, log_dir='./logs/cartpole-30', max_episode=4096)
    agent = A2CAgent(env, args, verbose=True)
    agent.train()


if __name__ == '__main__':
    main()
