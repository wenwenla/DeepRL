import gym
from deeprl_util.args import DDPGArgs
from deeprl_util.preprocessing import SimpleNormalizer
from model.actor import DDPGActor
from model.qnet import DDPGQNet
from ddpg import DDPGAgent


def train_ddpg():
    args = DDPGArgs()
    env = gym.make(args.env_name)
    agent = DDPGAgent(env, DDPGQNet, DDPGActor, SimpleNormalizer, args)
    for ep in range(args.max_ep):
        agent.train_one_episode()
        if ep % args.test_interval == 0:
            agent.test_model()


if __name__ == '__main__':
    train_ddpg()
