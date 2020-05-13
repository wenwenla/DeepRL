class DDPGArgs:

    def __init__(self):
        self.exp_cap = 1000000
        self.gamma = 0.95
        self.batch = 128
        self.test_interval = 32
        self.update_cnt = 200
        self.update_interval = 200
        self.actor_lr = 1e-4
        self.critic_lr = 5e-3

        self.env_name = input('input env name: ')
        self.action_bound = float(input('input action bound: '))
        self.max_ep = int(input('input max train episode: '))
        self.scale = float(input('input scale: '))

        self.log_dir = './logs/ddpg/{}'.format(self.env_name)
        self.save_path = './result/ddpg/{}'.format(self.env_name)


class DQNArgs:

    def __init__(self):
        self.exp_cap = 1000000
        self.gamma = 0.99
        self.batch = 128
        self.tau = 64
        self.max_ep = 2000
        self.log_interval = 1000
        self.test_interval = 32
        self.lr = 5e-4
        self.epsilon = 0.1
        self.env_name = 'LunarLander-v2'
        self.log_dir = './logs/dqn/{}'.format(self.env_name)
        self.save_dir = './result/dqn/{}'.format(self.env_name)


class DDQNArgs:

    def __init__(self):
        self.exp_cap = 1000000
        self.gamma = 0.99
        self.batch = 128
        self.tau = 64
        self.max_ep = 200000
        self.log_interval = 1000
        self.test_interval = 32
        self.lr = 5e-4
        self.epsilon = 0.1
        self.env_name = 'BipedalWalkerHardcore-v3'
        self.log_dir = './logs/ddqn/{}'.format(self.env_name)
        self.save_dir = './result/ddqn/{}'.format(self.env_name)
