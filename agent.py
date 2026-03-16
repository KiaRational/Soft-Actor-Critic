import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from model import GaussianPolicy, TwinQ
from replay_buffer import ReplayBuffer


class SAC:
    """
    Soft Actor-Critic v2 — twin Q-critics, automatic entropy tuning, no Value network.

    Interface matches the DQN codebase:
        act(obs)                              → (action, log_prob)
        process_transition(obs, r, term, trunc) → dict of losses
    """

    def __init__(
        self,
        actor:          GaussianPolicy,
        critic:         TwinQ,
        replay_buffer:  ReplayBuffer,
        lr:             float = 3e-4,
        lr_alpha:       float = 3e-4,    # separate slower lr for alpha — prevents early collapse
        gamma:          float = 0.99,
        tau:            float = 0.005,
        alpha:          float = 0.2,
        target_entropy: float = None,    # None = -act_dim (paper default); try -0.5*act_dim for more exploration
        batch_size:     int   = 256,
        start_steps:    int   = 10_000,
        reward_scale:   float = 1.0,
        device:         torch.device = None,
    ):
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.start_steps  = start_steps
        self.reward_scale = reward_scale
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Networks ────────────────────────────────────────────────────────
        self.actor          = actor.to(self.device)
        self.critic         = critic.to(self.device)
        self.critic_target  = deepcopy(critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Optimisers ──────────────────────────────────────────────────────
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # ── Automatic entropy tuning ─────────────────────────────────────────
        # -0.5 * act_dim keeps alpha from collapsing to near-zero early in training.
        # With -act_dim the policy log_prob drops below target immediately and alpha→0.
        self.target_entropy = target_entropy if target_entropy is not None else -0.5 * float(actor.mu.out_features)
        self.log_alpha      = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_opt      = optim.Adam([self.log_alpha], lr=lr_alpha)

        # ── Replay buffer ────────────────────────────────────────────────────
        self.replay_buffer = replay_buffer

        # ── State ────────────────────────────────────────────────────────────
        self.total_steps = 0
        self.last_obs    = None
        self.last_act    = None

    # ── Interaction ─────────────────────────────────────────────────────────

    def act(self, obs):
        """
        Returns (action np.ndarray, log_prob float).
        Should only be called after warmup — loop handles warmup externally.
        """
        assert self.total_steps >= self.start_steps, \
            "act() called during warmup — use env.action_space.sample() instead"

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, logp_t = self.actor.sample(obs_t)
        self.last_obs = obs
        self.last_act = action_t.cpu().numpy()[0]
        return self.last_act, logp_t.item()

    def process_transition(self, obs, reward, terminated, truncated):
        """
        Store transition and update. Returns critic_loss float (0.0 if no update yet).
        Matches DQN process_transition interface.
        """
        if self.last_obs is not None and self.last_act is not None:
            done = float(terminated)
            self.replay_buffer.store(
                self.last_obs, self.last_act,
                reward * self.reward_scale,
                obs, done,
            )
            self.total_steps += 1

            if self.total_steps >= self.start_steps and \
               self.replay_buffer.size >= self.batch_size:
                losses = self.update()
                return losses['critic_loss']

        return 0.0

    # ── Gradient update ──────────────────────────────────────────────────────

    def update(self):
        o, a, r, o2, d = self.replay_buffer.sample(self.batch_size, self.device)

        alpha = self.log_alpha.exp()

        # ── Critic loss ───────────────────────────────────────────────────────
        with torch.no_grad():
            a2, logp2    = self.actor.sample(o2)
            q1_t, q2_t   = self.critic_target(o2, a2)
            q_target     = torch.min(q1_t, q2_t) - alpha * logp2
            y            = r + self.gamma * (1 - d) * q_target

        q1, q2       = self.critic(o, a)
        critic_loss  = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor loss ────────────────────────────────────────────────────────
        a_new, logp  = self.actor.sample(o)
        q1_new, q2_new = self.critic(o, a_new)
        q_new        = torch.min(q1_new, q2_new)
        actor_loss   = (alpha.detach() * logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Alpha loss ────────────────────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── Polyak update ─────────────────────────────────────────────────────
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss':  actor_loss.item(),
            'alpha':       alpha.item(),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            'actor':     self.actor.state_dict(),
            'critic':    self.critic.state_dict(),
            'log_alpha': self.log_alpha.data,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target = deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():   # BUG FIX: was missing, caused grad computation through target
            p.requires_grad = False
        self.log_alpha.data.copy_(ckpt['log_alpha'])

    def set_to_eval_mode(self):
        self.actor.eval()

    def set_to_train_mode(self):
        self.actor.train()