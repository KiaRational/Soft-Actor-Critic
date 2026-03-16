"""
train_single.py — Train SAC for ONE seed, save results + checkpoint.

Usage (run each on separate GPU):
    python train_single.py --env Walker2d-v4 --seed 0
    python train_single.py --env Walker2d-v4 --seed 1
    python train_single.py --env Walker2d-v4 --seed 2

After all done:
    python plot_seeds.py --env Walker2d-v4 --input_dir results/walker
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym

import agent_environment
from model import GaussianPolicy, TwinQ
from replay_buffer import ReplayBuffer
from agent import SAC

torch.set_float32_matmul_precision("high")


def make_agent(obs_dim, act_dim, act_limit, args, device):
    actor  = GaussianPolicy(obs_dim, act_dim, act_limit, args.hidden)
    critic = TwinQ(obs_dim, act_dim, args.hidden)
    buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size)
    return SAC(
        actor         = actor,
        critic        = critic,
        replay_buffer = buffer,
        lr            = args.lr,
        lr_alpha      = args.lr_alpha,
        gamma         = args.gamma,
        tau           = args.tau,
        alpha         = args.alpha,
        batch_size    = args.batch_size,
        start_steps   = args.start_steps,
        reward_scale  = args.reward_scale,
        device        = device,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--env",          type=str,   required=True)
    p.add_argument("--seed",         type=int,   required=True)
    p.add_argument("--total_steps",  type=int,   default=1_000_000)
    p.add_argument("--start_steps",  type=int,   default=10_000)
    p.add_argument("--eval_every",   type=int,   default=5_000)
    p.add_argument("--eval_eps",     type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--buffer_size",  type=int,   default=1_000_000)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--lr_alpha",     type=float, default=3e-4)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--tau",          type=float, default=0.005)
    p.add_argument("--alpha",        type=float, default=0.2)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--output_dir",   type=str,   required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    tag = args.env.replace("-", "_").lower()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device} | Env: {args.env} | Seed: {seed}")

    _env = gym.make(args.env)
    obs_dim   = _env.observation_space.shape[0]
    act_dim   = _env.action_space.shape[0]
    act_limit = float(_env.action_space.high[0])
    _env.close()
    print(f"obs={obs_dim}  act={act_dim}  limit={act_limit}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env      = gym.make(args.env, render_mode=None)
    eval_env = gym.make(args.env, render_mode=None)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 1000)

    agent = make_agent(obs_dim, act_dim, act_limit, args, device)

    eval_returns, eval_timesteps, ep_returns, ep_timesteps = \
        agent_environment.agent_environment_step_loop(
            agent             = agent,
            env               = env,
            eval_env          = eval_env,
            num_steps         = args.total_steps,
            min_replay_size   = args.start_steps,
            eval_frequency    = args.eval_every,
            num_eval_episodes = args.eval_eps,
        )

    env.close()
    eval_env.close()

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, f"sac_{tag}_seed{seed}.pth")
    agent.save(ckpt_path)
    print(f"Checkpoint → {ckpt_path}")

    # Save results
    npz_path = os.path.join(args.output_dir, f"sac_{tag}_seed{seed}.npz")
    np.savez(npz_path,
        eval_returns=np.array(eval_returns),
        eval_timesteps=np.array(eval_timesteps),
        ep_returns=np.array(ep_returns),
        ep_timesteps=np.array(ep_timesteps),
        seed=seed,
        env=args.env,
    )
    print(f"Results → {npz_path}")
    print(f"Seed {seed} done | Final eval: {eval_returns[-1]:.1f}")