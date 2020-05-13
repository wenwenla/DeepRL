import os
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer


class DQNTransformer:

    EMPTY = np.zeros((210, 160), dtype=np.float32)

    def __init__(self):
        self._buffer = [
            DQNTransformer.EMPTY, 
            DQNTransformer.EMPTY,
            DQNTransformer.EMPTY,
            DQNTransformer.EMPTY
        ]
        self.shape = (4, 210, 160)

    def clear(self):
        self._buffer = self._buffer[:4]

    def add(self, state):
        state = np.dot(state[..., :3], [0.299, 0.587, 0.114])
        state /= 255.
        self._buffer.append(state)

    def get(self):
        return np.array([self._buffer[i] for i in [-4, -3, -2, -1]], dtype=np.float32)


class SimpleNormalizer:

    def __init__(self, env):
        self._env = env
        self._norm = Normalizer()
        self._norm.fit(self._gen_data(10000))

    def _gen_data(self, cap):
        data = []
        while len(data) < cap:
            data.append(self._env.reset())
            done = False
            while not done:
                action = self._env.action_space.sample()
                s, _, done, _ = self._env.step(action)
                data.append(s)
        return data

    def transform(self, state):
        res = self._norm.transform([state])[0]
        return res

    def save(self, path):
        fp = os.path.join(path, 'norm.pkl')
        with open(fp, 'wb') as f:
            pickle.dump(self._norm, f)

    def load(self, path):
        fp = os.path.join(path, 'norm.pkl')
        with open(fp, 'rb') as f:
            self._norm = pickle.load(f)
