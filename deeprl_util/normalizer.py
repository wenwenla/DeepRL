import numpy as np


class Normalizer:

    def __init__(self, env):
        state_dim = env.observation_space.shape[0]
        self._env = env
        self._mean = np.zeros((state_dim, ))
        self._std = np.zeros((state_dim, ))
        self._init()

    def _init(self):
        total = []
        state = self._env.reset()
        total.append(state)
        for _ in range(100000):
            action = self._env.action_space.sample()
            state_, _, done, _ = self._env.step(action)
            total.append(state_)
            state = state_
            if done:
                state = self._env.reset()
        self._mean = np.mean(total, 0)
        self._std = np.std(total, 0)
        self._std[self._std == 0] = 1.
        print('Normalizer Init Done...')

    def transform(self, state):
        return (state - self._mean) / self._std
