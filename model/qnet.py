import torch
import torch.nn as nn


class DDPGQNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self._fc0 = nn.Linear(state_dim + action_dim, 256)
        self._fc1 = nn.Linear(256, 128)
        self._fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self._fc0(x))
        x = torch.relu(self._fc1(x))
        x = self._fc2(x)
        return x
