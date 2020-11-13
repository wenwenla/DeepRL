import torch


class VNet(torch.nn.Module):

    def __init__(self, state_dim):
        super(VNet, self).__init__()
        self.fc0 = torch.nn.Linear(state_dim, 64)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class PolicyNet(torch.nn.Module):

    def __init__(self, state_dim, action_cnt):
        super(PolicyNet, self).__init__()
        self.fc0 = torch.nn.Linear(state_dim, 64)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, action_cnt)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
