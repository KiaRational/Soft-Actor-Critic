"""
run_ablation.py — Run ONE ablation config (any hyperparam) and save results.

Usage:
    python run_ablation.py --env Ant-v4 --study reward_scale --param reward_scale --value 5.0 --label RS5 --seeds 0 1 2
    python run_ablation.py --env Ant-v4 --study tau --param tau --value 0.001 --label tau0.001 --seeds 0 1 2
    python run_ablation.py --env Ant-v4 --study target_entropy --param target_entropy --value -4.0 --label H4 --seeds 0 1 2
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym

from model import GaussianPolicy, TwinQ
from replay_buffer import ReplayBuffer
from agent import SAC

torch.set_float32_matmul_precision("high")

DEFAULTS = dict(
    lr=3e-4, lr_alpha=3e-4, gamma=0.99, tau=0.005,
    alpha=0.2, target_entropy=None, batch_size=256,
    start_steps=10_000, reward_scale=1.0, hidden=256,
)


def make_agent(obs_dim, act_dim, act_limit, overrides, device):
    cfg = {**DEFAULTS, **overrides}
    actor  = GaussianPolicy(obs_dim, act_dim, act_limit, cfg['hidden'])
    critic = TwinQ(obs_dim, act_dim, cfg['hidden'])
    buffer = ReplayBuffer(obs_dim, act_dim, 1_000_000)
    return SAC(
        actor=actor, critic=critic, replay_buffer=buffer,
        lr=cfg['lr'], lr_alpha=cfg['lr_alpha'], gamma=cfg['gamma'],
        tau=cfg['tau'], alpha=cfg['alpha'],
        target_entropy=cfg['target_entropy'],
        batch_size=cfg['batch_size'], start_steps=cfg['start_steps'],
        reward_scale=cfg['reward_scale'], device=device,
    )


def run_one_seed(env_name, overrides, seed, total_steps, eval_every, eval_eps, device, output_dir, study, label):
    _env = gym.make(env_name)
    obs_dim = _env.observation_space.shape[0]
    act_dim = _env.action_space.shape[0]
    act_lim = float(_env.action_space.high[0])
    _env.close()

    torch.manual_seed(seed)
    np.random.seed(seed)

    env      = gym.make(env_name, render_mode=None)
    eval_env = gym.make(env_name, render_mode=None)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 1000)

    agent = make_agent(obs_dim, act_dim, act_lim, overrides, device)

    observation, _ = env.reset()
    episode_return = 0
    episode = 0

    eval_returns = []
    eval_steps = []
    alpha_at_eval = []

    for step in range(total_steps):
        if step < 10_000:
            action = env.action_space.sample()
            agent.last_obs = observation
            agent.last_act = action
        else:
            action, _ = agent.act(observation)

        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.process_transition(next_obs, reward, terminated, truncated)
        observation = next_obs
        episode_return += reward

        if terminated or truncated:
            alpha = agent.log_alpha.exp().item()
            if episode % 20 == 0:
                print(f"  Ep {episode:5d} | Step {step+1:>8,} | Ret {episode_return:>9.2f} | a {alpha:.4f}")
            episode_return = 0
            episode += 1
            observation, _ = env.reset()

        if step >= 10_000 and step % eval_every == 0:
            agent.set_to_eval_mode()
            evals = []
            for _ in range(eval_eps):
                e_obs, _ = eval_env.reset()
                e_ret, e_done = 0, False
                while not e_done:
                    obs_t = torch.tensor(e_obs, dtype=torch.float32).unsqueeze(0).to(device)
                    e_act = agent.actor.act(obs_t).cpu().numpy()[0]
                    e_obs, e_r, term, trunc, _ = eval_env.step(e_act)
                    e_ret += e_r
                    e_done = term or trunc
                evals.append(e_ret)
            agent.set_to_train_mode()

            mean_eval = np.mean(evals)
            cur_alpha = agent.log_alpha.exp().item()
            eval_returns.append(mean_eval)
            eval_steps.append(step)
            alpha_at_eval.append(cur_alpha)
            print(f"  >>> Eval @ {step+1:,} | Mean: {mean_eval:.2f} | a: {cur_alpha:.4f}")

    env.close()
    eval_env.close()

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"{study}_{label}_seed{seed}.pth")
    agent.save(ckpt_path)

    return np.array(eval_returns), np.array(eval_steps), np.array(alpha_at_eval)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--env",          type=str, required=True)
    p.add_argument("--study",        type=str, required=True, help="Study name: reward_scale, tau, target_entropy")
    p.add_argument("--param",        type=str, required=True, help="Param to override")
    p.add_argument("--value",        type=float, required=True)
    p.add_argument("--label",        type=str, required=True, help="Short label for filename")
    p.add_argument("--seeds",        nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--total_steps",  type=int, default=600_000)
    p.add_argument("--eval_every",   type=int, default=5_000)
    p.add_argument("--eval_eps",     type=int, default=10)
    p.add_argument("--output_dir",   type=str, default="results/ablation")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    overrides = {args.param: args.value}
    print(f"Device: {device}")
    print(f"Env: {args.env} | Study: {args.study} | {args.param}={args.value} | Label: {args.label}")
    print(f"Seeds: {args.seeds} | Steps: {args.total_steps:,}")

    all_returns = []
    all_alphas = []
    steps = None

    for seed in args.seeds:
        print(f"\n{'='*50} Seed {seed} {'='*50}")
        ret, stp, alp = run_one_seed(
            args.env, overrides, seed,
            args.total_steps, args.eval_every, args.eval_eps, device,
            args.output_dir, args.study, args.label
        )
        all_returns.append(ret)
        all_alphas.append(alp)
        steps = stp
        print(f"Seed {seed} done | Final eval: {ret[-1]:.1f}")

    min_len = min(len(r) for r in all_returns)
    returns_arr = np.array([r[:min_len] for r in all_returns])
    alphas_arr = np.array([a[:min_len] for a in all_alphas])
    steps_arr = steps[:min_len]

    path = os.path.join(args.output_dir, f"{args.study}_{args.label}.npz")
    np.savez(path,
        steps=steps_arr, returns=returns_arr, alphas=alphas_arr,
        param=args.param, value=args.value,
        env=args.env, study=args.study, label=args.label,
        seeds=np.array(args.seeds),
    )
    print(f"\nSaved -> {path}")
    print("Done.")