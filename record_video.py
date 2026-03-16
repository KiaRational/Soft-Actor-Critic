"""
record_video.py — Load a SAC checkpoint and record evaluation videos.

Usage:
    python record_video.py --env Walker2d-v4 --checkpoint results/sac_walker2d_v4_seed0.pth
    python record_video.py --env Ant-v4 --checkpoint results/sac_ant_v4_seed0.pth --episodes 5
    python record_video.py --env Ant-v4 --checkpoint results/sac_ant_v4_seed0.pth --output_dir videos

Requires: MUJOCO_GL=egl (set automatically for headless rendering)
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym

os.environ["MUJOCO_GL"] = "egl"

from model import GaussianPolicy, TwinQ


def load_agent(env_name, checkpoint_path, device):
    """Load environment dims and rebuild actor from checkpoint."""
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    env.close()

    actor = GaussianPolicy(obs_dim, act_dim, act_limit, hidden=256).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, act_limit={act_limit}")

    return actor, act_limit


def record_episodes(env_name, actor, device, n_episodes, output_dir, fps=30):
    """Record n episodes as mp4 videos."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        import imageio
    except ImportError:
        print("Installing imageio...")
        os.system("pip install imageio[ffmpeg] --quiet")
        import imageio

    env = gym.make(env_name, render_mode="rgb_array")
    tag = env_name.replace("-", "_").lower()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        frames = []
        done = False
        episode_return = 0
        steps = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor.act(obs_t).cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
            steps += 1
            frames.append(env.render())

        if not frames:
            continue

        # Try mp4 first, fall back to gif
        filename = f"sac_{tag}_ep{ep}.mp4"
        filepath = os.path.join(output_dir, filename)

        try:
            writer = imageio.get_writer(filepath, fps=fps, macro_block_size=1)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"  Episode {ep}: Return={episode_return:.1f}, Steps={steps} → {filepath}")
        except Exception as e:
            gif_path = filepath.replace(".mp4", ".gif")
            print(f"  MP4 failed ({e}), saving GIF...")
            imageio.mimsave(gif_path, frames[::2], fps=fps//2)  # skip frames for smaller gif
            print(f"  Episode {ep}: Return={episode_return:.1f}, Steps={steps} → {gif_path}")

    env.close()
    print(f"\nAll videos saved to: {output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Record SAC agent videos")
    p.add_argument("--env",        type=str, required=True, help="Gym environment name")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    p.add_argument("--episodes",   type=int, default=3,     help="Number of episodes to record")
    p.add_argument("--output_dir", type=str, default="videos")
    p.add_argument("--fps",        type=int, default=30)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    actor, act_limit = load_agent(args.env, args.checkpoint, device)
    record_episodes(args.env, actor, device, args.episodes, args.output_dir, args.fps)