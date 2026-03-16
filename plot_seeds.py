"""
plot_seeds.py — Read per-seed .npz files and generate assignment plot.

Usage:
    python plot_seeds.py --env Walker2d-v4 --input_dir results/walker
    python plot_seeds.py --env Ant-v4 --input_dir results/ant --title "(kghasemz) AntV4 - Soft Actor Critic"
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_seeds(input_dir, env_name):
    """Load all sac_{env}_seed*.npz files from directory."""
    tag = env_name.replace("-", "_").lower()
    seeds_data = []

    for fname in sorted(os.listdir(input_dir)):
        if fname.startswith(f"sac_{tag}_seed") and fname.endswith(".npz"):
            path = os.path.join(input_dir, fname)
            data = np.load(path, allow_pickle=True)
            seed = int(data['seed'])
            seeds_data.append({
                'seed': seed,
                'eval_returns': data['eval_returns'],
                'eval_timesteps': data['eval_timesteps'],
            })
            print(f"  Loaded: {fname} (seed={seed}, final_eval={data['eval_returns'][-1]:.1f})")

    if not seeds_data:
        print(f"ERROR: No files matching sac_{tag}_seed*.npz in {input_dir}")
        exit(1)

    seeds_data.sort(key=lambda x: x['seed'])
    return seeds_data


def plot(seeds_data, env_name, output_dir, title=None):
    # Align lengths
    min_len = min(len(s['eval_returns']) for s in seeds_data)
    steps = seeds_data[0]['eval_timesteps'][:min_len]
    y = np.array([s['eval_returns'][:min_len] for s in seeds_data])
    mean = y.mean(0)
    n_seeds = len(seeds_data)

    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Individual seeds as transparent lines
    for i, sd in enumerate(seeds_data):
        ax.plot(steps, y[i], lw=0.9, color=colors[i % len(colors)], alpha=0.3)

    # Bold mean
    ax.plot(steps, mean, lw=2.5, color='black', label=f'Mean (n={n_seeds})')

    # Legend with seed colors
    handles = []
    for i, sd in enumerate(seeds_data):
        handles.append(Line2D([0], [0], color=colors[i % len(colors)], lw=1.5, alpha=0.5,
                              label=f"Seed {sd['seed']}"))
    handles.append(Line2D([0], [0], color='black', lw=2.5, label=f'Mean (n={n_seeds})'))
    ax.legend(handles=handles, fontsize=11, loc='lower right')

    ax.set_xlabel('Time Steps', fontsize=13)
    ax.set_ylabel('Average Return', fontsize=13)

    if title is None:
        title = f"SAC — {env_name}"
    ax.set_title(title, fontsize=15, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.tight_layout()

    tag = env_name.replace("-", "_").lower()
    path = os.path.join(output_dir, f"sac_{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot → {path}")

    # Also save combined data
    np.savez(os.path.join(output_dir, f"sac_{tag}_combined.npz"),
             steps=steps, all_returns=y, mean=mean, std=y.std(0),
             seeds=np.array([s['seed'] for s in seeds_data]))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--env",       type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--title",     type=str, default=None)
    args = p.parse_args()

    print(f"Loading results for {args.env} from {args.input_dir}...")
    seeds_data = load_seeds(args.input_dir, args.env)
    plot(seeds_data, args.env, args.input_dir, args.title)
    print("Done.")