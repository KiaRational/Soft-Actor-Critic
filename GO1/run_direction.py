"""
run_direction.py
================
Evaluate a trained Go1 policy and save rollouts as MP4 videos.
Fully headless — uses MuJoCo's EGL offscreen renderer, no display needed.
Safe to run on Compute Canada compute nodes.

Usage
-----
# Single direction, 5 episodes
    python run_direction.py --checkpoint results/go1/sac_go1_seed0.pth \\
                            --direction "1,0,0" --num_episodes 5

# Random directions, one video per episode
    python run_direction.py --checkpoint results/go1/sac_go1_seed0.pth \\
                            --num_episodes 8

# Rotate through 8 directions and record each
    python run_direction.py --checkpoint results/go1/sac_go1_seed0.pth \\
                            --rotate_demo --num_directions 8

# On Compute Canada — make sure these are set before running:
#   module load mujoco/3.1.6
#   export MUJOCO_GL=egl
#   unset DISPLAY
"""

import os
import argparse
import numpy as np
import torch
import imageio          # pip install imageio[ffmpeg]

import go1_direction_env
from go1_direction_env import Go1DirectionEnv
from model         import GaussianPolicy, TwinQ
from replay_buffer import ReplayBuffer
from agent         import SAC


# ─────────────────────────────────────────────────────────────────────────────
# Policy loader
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(checkpoint_path, obs_dim, act_dim, act_limit, device, hidden=256):
    actor  = GaussianPolicy(obs_dim, act_dim, act_limit, hidden)
    critic = TwinQ(obs_dim, act_dim, hidden)
    buf    = ReplayBuffer(obs_dim, act_dim, size=1)   # dummy
    agent  = SAC(actor=actor, critic=critic, replay_buffer=buf, device=device)
    agent.load(checkpoint_path)
    agent.set_to_eval_mode()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Single episode rollout — returns frames + stats
# ─────────────────────────────────────────────────────────────────────────────

def rollout(env, agent, direction, device):
    """
    Run one episode with a fixed direction.
    Returns (frames: list[np.ndarray HxWx3], ep_return: float, n_steps: int)
    """
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    env.fixed_direction = direction

    obs, info = env.reset()
    frames = []
    ep_return = 0.0
    done = False
    step = 0

    while not done:
        # Inject direction into last 3 dims of observation
        obs[-3:] = direction

        obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.actor.act(obs_t).cpu().numpy()[0]

        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        done = terminated or truncated
        step += 1

        # Capture frame — rgb_array mode, no display needed
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    return frames, ep_return, step, info


# ─────────────────────────────────────────────────────────────────────────────
# Video writer
# ─────────────────────────────────────────────────────────────────────────────

def save_video(frames, path, fps=50):
    """Save a list of H×W×3 uint8 frames as an MP4."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                output_params=["-crf", "18"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {len(frames)} frames → {path}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Direction schedules
# ─────────────────────────────────────────────────────────────────────────────

def make_directions(args):
    """Return a list of (label, direction-array) tuples."""
    if args.direction:
        vals = [float(x) for x in args.direction.split(",")]
        d = np.array(vals, dtype=np.float32)
        d /= np.linalg.norm(d) + 1e-8
        return [(f"fixed_{d[0]:.2f}_{d[1]:.2f}_{d[2]:.2f}", d)] * args.num_episodes

    if args.rotate_demo:
        n = args.num_directions
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        dirs = []
        for i, a in enumerate(angles):
            d = np.array([np.cos(a), np.sin(a), 0.0], dtype=np.float32)
            dirs.append((f"dir{i:02d}_angle{np.degrees(a):.0f}deg", d))
        dirs = dirs * (args.num_episodes // n + 1)
        return dirs[:args.num_episodes]

    # Random directions
    rng = np.random.default_rng(args.seed)
    dirs = []
    for i in range(args.num_episodes):
        if args.direction_2d:
            a = rng.uniform(0, 2 * np.pi)
            d = np.array([np.cos(a), np.sin(a), 0.0], dtype=np.float32)
        else:
            d = rng.standard_normal(3).astype(np.float32)
            d /= np.linalg.norm(d) + 1e-8
        dirs.append((f"ep{i:02d}_rand", d))
    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# Optional frame annotation (requires Pillow)
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frames(frames, direction, ep_return, n_steps):
    """Burn direction vector and step count into top-left of each frame."""
    try:
        from PIL import Image, ImageDraw
        annotated = []
        for i, frame in enumerate(frames):
            img  = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            text = (
                f"dir=[{direction[0]:+.2f},{direction[1]:+.2f},{direction[2]:+.2f}]\n"
                f"step {i+1}/{n_steps}  return={ep_return:.1f}"
            )
            draw.text((11, 11), text, fill=(0, 0, 0))      # shadow
            draw.text((10, 10), text, fill=(255, 255, 255)) # text
            annotated.append(np.array(img))
        return annotated
    except ImportError:
        return frames   # Pillow not installed — skip annotation


# ─────────────────────────────────────────────────────────────────────────────
# Theoretical maximum return
# ─────────────────────────────────────────────────────────────────────────────

def theoretical_max(env):
    """
    Per-step upper bound (ideal agent, no penalty terms):

        r_vel_max  = W_VEL  * v_max   = 2.0 * 3.5 = 7.0
        r_alive_max = W_ALIVE * 1.0   = 1.0 * 1.0 = 1.0
        r_lat      = 0   (zero lateral drift)
        r_rot      = 0   (zero yaw rate)
        r_torque   ≈ 0   (negligible at light loads)

    r_max_per_step = 8.0
    R_max = 8.0 * MAX_STEPS = 8.0 * 1000 = 8000

    In practice a well-trained policy achieves ~3000–5000.
    The Go1 physical top speed is ~3.5 m/s; comfortable trot is ~1.5 m/s.
    """
    v_max          = 3.5   # Go1 physical top speed (m/s)
    r_max_per_step = env.W_VEL * v_max + env.W_ALIVE
    return r_max_per_step * env.MAX_STEPS


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build env — rgb_array = EGL offscreen, no X11 needed ─────────────────
    env = Go1DirectionEnv(
        mjcf_path       = args.mjcf_path,
        render_mode     = "rgb_array",
        fixed_direction = np.array([1.0, 0.0, 0.0]),   # overridden per episode
        direction_2d    = args.direction_2d,
    )

    obs_dim   = env.observation_space.shape[0]
    act_dim   = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    print(f"obs={obs_dim}  act={act_dim}  limit={act_limit}")
    print(f"Theoretical max return: {theoretical_max(env):.0f}")

    # ── Load policy ───────────────────────────────────────────────────────────
    agent = load_policy(args.checkpoint, obs_dim, act_dim, act_limit,
                        device, args.hidden)

    # ── Direction schedule ────────────────────────────────────────────────────
    directions = make_directions(args)
    os.makedirs(args.video_dir, exist_ok=True)
    print(f"\nRecording {len(directions)} episode(s) → {args.video_dir}/\n")

    returns = []

    for ep_idx, (label, direction) in enumerate(directions):
        d = direction
        print(f"Ep {ep_idx+1:3d}/{len(directions)} | "
              f"dir=[{d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}] | {label}")

        frames, ep_return, n_steps, _ = rollout(env, agent, direction, device)
        returns.append(ep_return)

        if not args.no_annotation:
            frames = annotate_frames(frames, direction, ep_return, n_steps)

        video_name = f"ep{ep_idx+1:03d}_{label}_ret{ep_return:.0f}.mp4"
        save_video(frames, os.path.join(args.video_dir, video_name), fps=args.fps)
        print(f"  steps={n_steps:4d}  return={ep_return:7.2f}")

    tmax = theoretical_max(env)  # compute BEFORE close
    env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Episodes         : {len(returns)}")
    print(f"  Mean return      : {np.mean(returns):7.2f}")
    print(f"  Std  return      : {np.std(returns):7.2f}")
    print(f"  Best return      : {np.max(returns):7.2f}")
    print(f"  Worst return     : {np.min(returns):7.2f}")
    print(f"  Theoretical max  : {tmax:7.2f}")
    print(f"{'='*55}")

    # Save stats CSV
    stats_path = os.path.join(args.video_dir, "eval_stats.csv")
    with open(stats_path, "w") as f:
        f.write("episode,label,dir_x,dir_y,dir_z,return\n")
        for i, ((lbl, d), r) in enumerate(zip(directions, returns)):
            f.write(f"{i+1},{lbl},{d[0]:.4f},{d[1]:.4f},{d[2]:.4f},{r:.4f}\n")
    print(f"  Stats CSV → {stats_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Headless Go1 evaluation — saves MP4 videos, no display needed"
    )

    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--mjcf_path",      type=str, default=go1_direction_env.MJCF_PATH)
    p.add_argument("--direction_2d",   action="store_true")
    p.add_argument("--hidden",         type=int, default=256)

    # Direction schedule — pick one
    p.add_argument("--direction",      type=str, default=None,
                   help="Fixed 'x,y,z' direction, e.g. '1,0,0'")
    p.add_argument("--rotate_demo",    action="store_true",
                   help="Rotate through evenly-spaced horizontal directions")
    p.add_argument("--num_directions", type=int, default=8)

    p.add_argument("--num_episodes",   type=int, default=5)
    p.add_argument("--video_dir",      type=str, default="videos/go1")
    p.add_argument("--fps",            type=int, default=50,
                   help="Must match env DT (50 Hz = 1/0.02 s)")
    p.add_argument("--no_annotation",  action="store_true",
                   help="Skip text overlay (no Pillow needed)")
    p.add_argument("--seed",           type=int, default=42)

    args = p.parse_args()
    main(args)