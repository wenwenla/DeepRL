import torch
import torch.nn as nn


class DDPGActor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self._fc0 = nn.Linear(state_dim, 128)
        self._fc1 = nn.Linear(128, 64)
        self._fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self._fc0(x))
        x = torch.relu(self._fc1(x))
        x = self._fc2(x)
        return torch.tanh(x)
