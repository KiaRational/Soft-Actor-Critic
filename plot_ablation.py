"""
plot_ablation.py — Plot ablation results grouped by study.

Produces one plot per study (reward_scale, tau, target_entropy)
matching the paper style: mean line + fill_between shading.

Usage:
    python plot_ablation.py
    python plot_ablation.py --input_dir results/ablation
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def smooth(y, window=5):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='same')


def load_all(input_dir):
    by_study = defaultdict(list)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.npz'):
            continue
        path = os.path.join(input_dir, fname)
        try:
            data = np.load(path, allow_pickle=True)
        except:
            continue
        study = str(data['study']) if 'study' in data else None
        if study is None:
            continue
        entry = {
            'steps':   data['steps'],
            'returns': data['returns'],
            'alphas':  data['alphas'],
            'label':   str(data['label']),
            'value':   float(data['value']),
            'param':   str(data['param']),
            'env':     str(data['env']),
        }
        by_study[study].append(entry)
        print(f"  Loaded: {fname} -> study={study}, label={entry['label']}, "
              f"value={entry['value']}, seeds={entry['returns'].shape[0]}")
    for study in by_study:
        by_study[study].sort(key=lambda r: r['value'])
    return dict(by_study)


def plot_study(results_list, study_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, res in enumerate(results_list):
        x = res['steps'] / 1e6
        y = res['returns']
        color = COLORS[i % len(COLORS)]

        mean_r = smooth(y.mean(axis=0))
        min_r = y.min(axis=0)
        max_r = y.max(axis=0)

        val = res['value']
        label = f"{int(val)}" if val == int(val) else f"{val}"

        ax.plot(x, mean_r, lw=2, color=color, label=label)
        ax.fill_between(x, min_r, max_r, alpha=0.15, color=color)

    param_display = {
        'reward_scale':   '(kghasemz) Reward Scale',
        'tau':            r'(kghasemz) Target Smoothing Coefficient ($\tau$)',
        'target_entropy': r'(kghasemz) Target Entropy ($\bar{H}$)',
    }

    ax.set_xlabel('million steps', fontsize=12)
    ax.set_ylabel('average return', fontsize=12)
    ax.set_title('(kghasemz) ' + param_display.get(study_name, study_name), fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f"ablation_{study_name}.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


def plot_combined_panel(all_studies, output_dir):
    studies = list(all_studies.keys())
    n = len(studies)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5))
    if n == 1:
        axes = [axes]

    param_display = {
        'reward_scale':   '(kghasemz) Reward Scale',
        'tau':            r'(kghasemz) Target Smoothing Coefficient ($\tau$)',
        'target_entropy': r'(kghasemz) Target Entropy ($\bar{H}$)',
    }
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

    for idx, (study_name, results_list) in enumerate(all_studies.items()):
        ax = axes[idx]
        for i, res in enumerate(results_list):
            x = res['steps'] / 1e6
            y = res['returns']
            color = COLORS[i % len(COLORS)]
            mean_r = smooth(y.mean(axis=0))
            min_r = y.min(axis=0)
            max_r = y.max(axis=0)
            val = res['value']
            label = f"{int(val)}" if val == int(val) else f"{val}"
            ax.plot(x, mean_r, lw=2, color=color, label=label)
            ax.fill_between(x, min_r, max_r, alpha=0.15, color=color)

        title = param_display.get(study_name, study_name)
        if not title.startswith('(kghasemz)'):
            title = '(kghasemz) ' + title
        ax.set_xlabel('million steps', fontsize=11)
        ax.set_ylabel('average return', fontsize=11)
        ax.set_title(f"{panel_labels[idx]} {title}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_all_studies.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


def write_summary(all_studies, output_dir):
    lines = []
    lines.append(f"{'Study':<18} {'Label':<12} {'Value':<10} {'Final Eval (mean+-std)':<25} {'Final Alpha'}")
    lines.append("-" * 80)
    for study, results in all_studies.items():
        for res in results:
            chunk = res['returns'][:, -10:]
            per_seed = chunk.mean(axis=1)
            f_alpha = res['alphas'][:, -5:].mean()
            lines.append(
                f"{study:<18} {res['label']:<12} {res['value']:<10.4f} "
                f"{per_seed.mean():>8.1f} +- {per_seed.std():>5.1f}          "
                f"{f_alpha:.4f}"
            )
    txt = "\n".join(lines)
    print("\n" + txt)
    path = os.path.join(output_dir, "ablation_summary.txt")
    with open(path, 'w') as f:
        f.write(txt + "\n")
    print(f"\n  -> {path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  type=str, default="results/ablation")
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()
    out = args.output_dir or args.input_dir
    os.makedirs(out, exist_ok=True)
    print("Loading .npz files...")
    all_studies = load_all(args.input_dir)
    if not all_studies:
        print("ERROR: No .npz files found in", args.input_dir)
        exit(1)
    print(f"\nFound {len(all_studies)} studies: {list(all_studies.keys())}\n")
    print("Generating plots...")
    for study_name, results_list in all_studies.items():
        plot_study(results_list, study_name, out)
    if len(all_studies) >= 2:
        plot_combined_panel(all_studies, out)
    write_summary(all_studies, out)
    print(f"\nAll outputs in: {out}/")