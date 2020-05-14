import os
import pickle
import numpy as np
import PIL.Image as Image
from sklearn.preprocessing import Normalizer


class Transormer:

    def __init__(self, env):
        pass

    def reset(self):
        pass

    def transform(self, state):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class DQNTransformer(Transormer):

    WIDTH = 64
    HEIGHT = 64
    EMPTY = np.zeros((WIDTH, HEIGHT), dtype=np.float32)

    def __init__(self, env):
        super().__init__(env)
        self._buffer = [
            DQNTransformer.EMPTY,
            DQNTransformer.EMPTY,
            DQNTransformer.EMPTY,
            DQNTransformer.EMPTY
        ]

    def reset(self):
        self._buffer = self._buffer[:4]

    def transform(self, state):
        img = Image.fromarray(state)
        img.save('./origin.jpg')
        img = img.convert('L')
        img = img.crop((0, 34, 160, 194))
        img.save('./crop.jpg')
        img = img.resize((DQNTransformer.HEIGHT, DQNTransformer.WIDTH))
        img.save('./resize.jpg')
        # state = np.dot(state[..., :3], [0.299, 0.587, 0.114])
        # img = Image.fromarray(state)
        # img.save('./img.jpg')
        # res = img.resize((DQNTransformer.HEIGHT, DQNTransformer.WIDTH), Image.ANTIALIAS)
        # # res.save('./img.jpg')
        # res = np.asarray(res).copy()
        
        # res /= 255.
        # self._buffer.append(res)
        return np.array([self._buffer[i] for i in [-4, -3, -2, -1]], dtype=np.float32)


class SimpleNormalizer(Transormer):

    def __init__(self, env):
        super().__init__()
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
