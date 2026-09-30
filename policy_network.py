import torch
import torch.nn as nn
import torch.nn.functional as F

N = 4
state_dim = 4
num_actions = 5

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action_dist(self, state_tensor):
        logits = self.forward(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        return dist

    def select_action(self, state_tensor):
        dist = self.get_action_dist(state_tensor)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx.item(), log_prob


def encode_state(predator_pos, prey_pos, N):
    dp = predator_pos
    pp = prey_pos
    state_vec = torch.tensor([
        (dp[0] - 1) / (N - 1) if N > 1 else 0.0,
        (dp[1] - 1) / (N - 1) if N > 1 else 0.0,
        (pp[0] - 1) / (N - 1) if N > 1 else 0.0,
        (pp[1] - 1) / (N - 1) if N > 1 else 0.0,
    ], dtype=torch.float32)
    return state_vec


action_map = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
