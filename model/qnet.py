import torch
import torch.nn as nn


class DDPGQNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        assert len(state_dim) == 1
        state_dim = state_dim[0]
        self._fc0 = nn.Linear(state_dim + action_dim, 256)
        self._fc1 = nn.Linear(256, 128)
        self._fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self._fc0(x))
        x = torch.relu(self._fc1(x))
        x = self._fc2(x)
        return x


class DQNQNet(nn.Module):

    def __init__(self, state_shape, action_cnt):
        super().__init__()
        w, h = state_shape[1], state_shape[2]
        self.conv0 = nn.Conv2d(state_shape[0], 16, (8, 8), 4)
        w //= 4; w -= 1; h //= 4; h -= 1
        self.conv1 = nn.Conv2d(16, 32, (4, 4), 2)
        w //= 2; w -= 1; h //= 2; h -= 1
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(w * h * 32, action_cnt)

    def forward(self, x):
        x = torch.relu(self.conv0(x))
        x = torch.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x
