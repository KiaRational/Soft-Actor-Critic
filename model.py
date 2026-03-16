import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX =  2


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Squashed Gaussian actor."""

    def __init__(self, obs_dim, act_dim, act_limit, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden], output_activation=nn.ReLU)
        self.mu        = nn.Linear(hidden, act_dim)
        self.log_std   = nn.Linear(hidden, act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        x       = self.net(obs)
        mu      = self.mu(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        """Stochastic action + log prob with numerically stable tanh correction."""
        mu, log_std = self(obs)
        std  = log_std.exp()
        dist = Normal(mu, std)
        u    = dist.rsample()
        a    = torch.tanh(u)
        logp = dist.log_prob(u).sum(-1, keepdim=True)
        logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1, keepdim=True)
        return a * self.act_limit, logp

    @torch.no_grad()
    def act(self, obs):
        """Deterministic greedy action for evaluation."""
        mu, _ = self(obs)
        return torch.tanh(mu) * self.act_limit


class TwinQ(nn.Module):
    """Twin Q-networks."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)