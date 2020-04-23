class DDPGArgs:

    def __init__(self):
        # TODO
        self.exp_cap = 1000
        self.gamma = 0.95
        self.batch = 128
        self.test_interval = 32
        self.update_cnt = 200
        self.update_interval = 200
        self.actor_lr = 1e-4
        self.critic_lr = 5e-3
        self.epsilon = 0.1

        self.env_name = input('input env name: ')
        self.action_bound = float(input('input action bound: '))
        self.max_ep = int(input('input max train episode: '))
        self.log_dir = input('input log dir: ')
        self.save_path = input('input save dir: ')
        self.scale = float(input('input scale: '))


class DQNArgs:

    def __init__(self):
        self.exp_cap = 500
        self.gamma = 0.95
        self.batch = 32
        self.max_ep = 100000
        self.log_interval = 1000
        self.test_interval = 100
        self.critic_lr = 5e-3
        self.epsilon = 0.1
        self.env_name = 'Breakout-v0'
        self.log_dir = './logs/dqn/Breakout-v0'
        self.save_dir = './result/dqn/Breakout-v0'
