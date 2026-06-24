import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def log(msg): print(msg, flush=True)

log(f"[startup] Python {sys.version.split()[0]}  PID={os.getpid()}")
log(f"[startup] CWD={os.getcwd()}")
log("[startup] Importing...")

import argparse;     log("[import] argparse OK")
import numpy as np;  log("[import] numpy OK")
import torch;        log(f"[import] torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"[import] GPU: {torch.cuda.get_device_name(0)}")

from model         import GaussianPolicy, TwinQ;  log("[import] model OK")
from replay_buffer import ReplayBuffer;            log("[import] replay_buffer OK")
from agent         import SAC;                     log("[import] agent OK")
from agent_environment import agent_environment_step_loop; log("[import] agent_environment OK")
import go1_direction_env;                          log("[import] go1_direction_env OK")
log("[startup] All imports done.\n")

torch.set_float32_matmul_precision("high")


def make_env(args, render_mode=None):
    fixed_dir = None
    if args.fixed_direction:
        vals = [float(x) for x in args.fixed_direction.split(",")]
        fixed_dir = np.array(vals, dtype=np.float32)
        fixed_dir /= np.linalg.norm(fixed_dir) + 1e-8
    return go1_direction_env.Go1DirectionEnv(
        mjcf_path       = args.mjcf_path,
        render_mode     = render_mode,
        fixed_direction = fixed_dir,
        direction_2d    = args.direction_2d,
    )


def make_agent(obs_dim, act_dim, act_limit, args, device):
    actor  = GaussianPolicy(obs_dim, act_dim, act_limit, args.hidden)
    critic = TwinQ(obs_dim, act_dim, args.hidden)
    buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size)
    return SAC(
        actor=actor, critic=critic, replay_buffer=buffer,
        lr=args.lr, lr_alpha=args.lr_alpha, gamma=args.gamma,
        tau=args.tau, alpha=args.alpha, batch_size=args.batch_size,
        start_steps=args.start_steps, reward_scale=args.reward_scale,
        target_entropy=args.target_entropy,
        device=device,
    )


def train(args, device):
    os.makedirs(args.output_dir, exist_ok=True)

    log("=" * 60)
    log(f"  SAC Go1 Direction Training")
    log(f"  Seed          : {args.seed}")
    log(f"  Device        : {device}")
    log(f"  Total steps   : {args.total_steps:,}")
    log(f"  Warmup steps  : {args.start_steps:,}")
    log(f"  Batch size    : {args.batch_size}")
    log(f"  LR            : {args.lr}  LR_alpha: {args.lr_alpha}")
    log(f"  Alpha init    : {args.alpha}")
    log(f"  Target entropy: {args.target_entropy}")
    log(f"  Reward scale  : {args.reward_scale}")
    log(f"  Fixed dir     : {args.fixed_direction}")
    log(f"  direction_2d  : {args.direction_2d}")
    log(f"  Output dir    : {args.output_dir}")
    log("=" * 60 + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log("[train] Building environments...")
    env      = make_env(args)
    eval_env = make_env(args)
    obs_dim   = env.observation_space.shape[0]
    act_dim   = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    log(f"[train] obs={obs_dim}  act={act_dim}  limit={act_limit}")

    env.action_space.seed(args.seed)
    eval_env.action_space.seed(args.seed + 1000)

    log("[train] Building agent...")
    agent = make_agent(obs_dim, act_dim, act_limit, args, device)

    # Log target entropy that was actually set
    log(f"[train] target_entropy={agent.target_entropy:.4f}  (act_dim={act_dim})")
    log(f"[train] Starting step loop via agent_environment_step_loop...\n")

    # Use the ORIGINAL agent_environment loop — it correctly uses
    # agent.process_transition() which handles buffer storage + updates
    # in the right order with the correct total_steps counter.
    eval_returns, eval_timesteps, ep_returns, ep_timesteps = \
        agent_environment_step_loop(
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

    # Save
    tag       = f"go1_seed{args.seed}"
    ckpt_path = os.path.join(args.output_dir, f"sac_{tag}.pth")
    npz_path  = os.path.join(args.output_dir, f"sac_{tag}.npz")

    agent.save(ckpt_path)
    log(f"\nCheckpoint saved -> {ckpt_path}")

    np.savez(npz_path,
        eval_returns   = np.array(eval_returns),
        eval_timesteps = np.array(eval_timesteps),
        ep_returns     = np.array(ep_returns),
        ep_timesteps   = np.array(ep_timesteps),
        seed           = args.seed,
    )
    log(f"Results saved   -> {npz_path}")
    if eval_returns:
        log(f"Final eval return: {eval_returns[-1]:.2f}")
    log(f"\nSeed {args.seed} DONE.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf_path",       type=str,   default=go1_direction_env.MJCF_PATH)
    p.add_argument("--direction_2d",    action="store_true")
    p.add_argument("--fixed_direction", type=str,   default=None)
    p.add_argument("--seed",            type=int,   required=True)
    p.add_argument("--total_steps",     type=int,   default=3_000_000)
    p.add_argument("--start_steps",     type=int,   default=10_000)
    p.add_argument("--eval_every",      type=int,   default=10_000)
    p.add_argument("--eval_eps",        type=int,   default=5)
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--buffer_size",     type=int,   default=1_000_000)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--lr_alpha",        type=float, default=3e-4)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--tau",             type=float, default=0.005)
    p.add_argument("--alpha",           type=float, default=0.2)
    p.add_argument("--reward_scale",    type=float, default=1.0)
    p.add_argument("--hidden",          type=int,   default=256)
    p.add_argument("--target_entropy",  type=float, default=None)
    p.add_argument("--output_dir",      type=str,   default="results/go1")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)