import time
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model.actor import DDPGActor as Actor
from model.qnet import DDPGQNet as QNet
from deeprl_util.buffer import ReplayBuffer
from deeprl_util.normalizer import Normalizer
from deeprl_util.args import Args


class DDPGAgent:

    def __init__(self, env, args):
        self.state_dim, self.action_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self._args = args
        self._actor = Actor(self.state_dim, self.action_dim)
        self._critic = QNet(self.state_dim, self.action_dim)
        self._target_actor = Actor(self.state_dim, self.action_dim)
        self._target_actor.load_state_dict(self._actor.state_dict())
        self._target_critic = QNet(self.state_dim, self.action_dim)
        self._target_critic.load_state_dict(self._critic.state_dict())
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=self._args.actor_lr)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=self._args.critic_lr)

        self._critic_loss_fn = torch.nn.MSELoss()

        self._exp = ReplayBuffer(args.exp_cap, self.state_dim, self.action_dim)
        self._env = env
        self._norm = Normalizer(env)

        self._update_cnt = 0
        self._train_ep = 0
        self._update_tick = self._args.update_interval
        self._sw = SummaryWriter(self._args.log_dir)

    def choose_action_with_exploration(self, state):
        action = self.choose_action(state)
        noise = np.random.normal(0, self._args.scale, (self.action_dim, ))
        action = np.clip(action + noise, -1, 1) # clip action between [-1, 1]
        return action

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            action = self._actor(state)
        action = action.detach().numpy()
        return action

    def soft_copy_parms(self):
        with torch.no_grad():
            for t, s in zip(self._target_actor.parameters(), self._actor.parameters()):
                t.copy_(0.95 * t.data + 0.05 * s.data)
            for t, s in zip(self._target_critic.parameters(), self._critic.parameters()):
                t.copy_(0.95 * t.data + 0.05 * s.data)

    def update(self):
        self._update_tick -= 1
        if self._update_tick != 0:
            return
        self._update_tick = self._args.update_interval

        for _ in range(self._args.update_cnt):
            samples = self._exp.sample(self._args.batch)
            s, a, r, s_, d = samples

            # update critic
            with torch.no_grad():
                opt_a = self._target_actor(torch.Tensor(s_)).detach().numpy()
                target_critic_input = np.hstack((s_, opt_a))
                target_critic_output = self._target_critic(torch.FloatTensor(target_critic_input))
                target_critic_output[d] = 0
                target_critic_output *= self._args.gamma
                target_critic_output += r.reshape((-1, 1))
                target_critic_output = target_critic_output.float()

            critic_output = self._critic(torch.FloatTensor(np.hstack((s, a))))
            critic_loss = self._critic_loss_fn(critic_output, target_critic_output)
            self._critic_optim.zero_grad()
            critic_loss.backward()
            self._critic_optim.step()
            # finished

            # update actor, maximize Q((s, actor_output(s))
            opt_a = self._actor(torch.FloatTensor(s))
            q_input = torch.cat([torch.FloatTensor(s), opt_a], 1)
            q_val = self._target_critic(torch.FloatTensor(q_input))
            actor_loss = -q_val.mean()
            self._actor_optim.zero_grad()
            actor_loss.backward()
            self._actor_optim.step()
            # finished

            # copy parms to target
            self.soft_copy_parms()
            self._update_cnt += 1

            # log loss
            self._sw.add_scalar('loss/actor', actor_loss.detach().item(), self._update_cnt)
            self._sw.add_scalar('loss/critic', critic_loss.detach().item(), self._update_cnt)

    def train_one_episode(self):
        state = self._env.reset()
        done = False
        total_reward = 0
        while not done:
            action = self.choose_action_with_exploration(self._norm.transform(state))
            state_, reward, done, _ = self._env.step(action * self._args.action_bound)
            self._exp.add(
                self._norm.transform(state),
                action,
                reward,
                self._norm.transform(state_),
                done
            )
            self.update()
            state = state_
            total_reward += reward
        self._train_ep += 1
        # log train_reward
        # self._norm.debug()

        self._sw.add_scalar('step_reward/train', total_reward, self._update_cnt)
        self._sw.add_scalar('ep_reward/train', total_reward, self._train_ep)
        return total_reward

    def test_one_episode(self, render=False):
        state = self._env.reset()
        if render:
            self._env.render()
            time.sleep(0.1)
        done = False
        total_reward = 0
        while not done:
            action = self.choose_action(self._norm.transform(state))
            state_, reward, done, _ = self._env.step(action * self._args.action_bound)
            if render:
                self._env.render()
                time.sleep(0.1)
            state = state_
            total_reward += reward
        return total_reward

    def test_model(self, cnt=10):
        r = [self.test_one_episode() for _ in range(cnt)]
        r_mean = np.mean(r)
        self._sw.add_scalar('step_reward/test', r_mean, self._update_cnt)
        self._sw.add_scalar('ep_reward/test', r_mean, self._train_ep)
        return r_mean

    def load(self, name):
        s = torch.load(name)
        self._actor.load_state_dict(s)

    def save(self, path):
        path = os.path.join(path, 'best.pkl')
        torch.save(self._actor.state_dict(), path)


def train_ddpg():
    import gym
    args = Args()
    prev = -1e9
    env = gym.make(args.env_name)
    ddpg_agent = DDPGAgent(env, args)
    for ep in range(args.max_ep):
        ddpg_agent.train_one_episode()
        if ep % args.test_interval == 0:
            r = ddpg_agent.test_model()
            if r > prev:
                prev = r
                ddpg_agent.save(args.save_path)


if __name__ == '__main__':
    train_ddpg()
