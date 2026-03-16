import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-size circular replay buffer backed by pre-allocated numpy arrays.
    Much faster than a deque of dicts because:
      - No Python object overhead per transition
      - Batch sampling is a single array slice, not a Python loop
    """

    def __init__(self, obs_dim, act_dim, size=1_000_000):
        self.obs      = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act      = np.zeros((size, act_dim), dtype=np.float32)
        self.rew      = np.zeros((size, 1),       dtype=np.float32)
        self.done     = np.zeros((size, 1),       dtype=np.float32)
        self.ptr  = 0
        self.size = 0
        self.max  = size

    def __len__(self):
        return self.size

    def store(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.act[self.ptr]      = act
        self.rew[self.ptr]      = rew
        self.done[self.ptr]     = done
        self.ptr  = (self.ptr + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        # from_numpy is zero-copy on CPU side — faster than torch.tensor()
        return (
            torch.from_numpy(self.obs[idx]).to(device),
            torch.from_numpy(self.act[idx]).to(device),
            torch.from_numpy(self.rew[idx]).to(device),
            torch.from_numpy(self.next_obs[idx]).to(device),
            torch.from_numpy(self.done[idx]).to(device),
        )