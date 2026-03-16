"""
merge_seeds.py — Merge per-seed ablation .npz files into combined files.

When seeds are run as separate jobs, each saves its own .npz like:
    ant_target_entropy_H8_s0.npz
    ant_target_entropy_H8_s1.npz

This script merges them into:
    ant_target_entropy_H8.npz  (with both seeds combined)

Run before plot_ablation.py:
    python merge_seeds.py
    python plot_ablation.py
"""

import os
import numpy as np
from collections import defaultdict

ablation_dir = "results/ablation"

# Group files by base name (strip _s0, _s1 suffix)
groups = defaultdict(list)

for f in sorted(os.listdir(ablation_dir)):
    if not f.endswith('.npz'):
        continue
    # Only process seed-split files
    if '_s0' not in f and '_s1' not in f and '_s2' not in f:
        continue
    # Get base name by removing seed suffix
    base = f
    for suffix in ['_s0.npz', '_s1.npz', '_s2.npz']:
        base = base.replace(suffix, '')
    groups[base].append(f)

if not groups:
    print("No seed-split .npz files found. Nothing to merge.")
    print("(Looking for files with _s0, _s1, _s2 in results/ablation/)")
    exit(0)

for base, files in sorted(groups.items()):
    print(f"\nMerging: {files}")
    
    seeds_data = []
    for f in sorted(files):
        path = os.path.join(ablation_dir, f)
        d = np.load(path, allow_pickle=True)
        seeds_data.append(d)
    
    # Align lengths across seeds
    min_len = min(len(d['returns'][0]) for d in seeds_data)
    
    # Stack returns and alphas from all seed files
    returns = np.concatenate([d['returns'][:, :min_len] for d in seeds_data], axis=0)
    alphas = np.concatenate([d['alphas'][:, :min_len] for d in seeds_data], axis=0)
    steps = seeds_data[0]['steps'][:min_len]
    
    # Clean up label (remove _s0, _s1)
    d0 = seeds_data[0]
    label = str(d0['label']).replace('_s0', '').replace('_s1', '').replace('_s2', '')
    
    # Save merged file
    out_path = os.path.join(ablation_dir, f"{base}.npz")
    np.savez(out_path,
        steps=steps,
        returns=returns,
        alphas=alphas,
        param=str(d0['param']),
        value=float(d0['value']),
        env=str(d0['env']),
        study=str(d0['study']),
        label=label,
        seeds=np.arange(returns.shape[0]),
    )
    print(f"  -> {out_path} (seeds={returns.shape[0]}, evals={returns.shape[1]})")

print("\nDone. Now run: python plot_ablation.py")