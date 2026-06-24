import numpy as np
import torch


def agent_environment_step_loop(
    agent,
    env,
    eval_env,                    # separate env for eval — never touches training state
    num_steps,
    min_replay_size,
    eval_frequency=5000,
    num_eval_episodes=10,
    debug=False,
):
    observation, info = env.reset()
    episode_returns      = []
    episodes_timesteps   = []
    evaluation_returns   = []
    evaluation_timesteps = []
    episode_return        = 0
    global_running_reward = 0
    episode               = 0

    for step in range(num_steps):

        # warmup: random actions until replay buffer is filled
        if step < min_replay_size:
            action = env.action_space.sample()
            agent.last_obs = observation
            agent.last_act = action
        else:
            action, _ = agent.act(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.process_transition(next_observation, reward, terminated, truncated)

        observation    = next_observation
        episode_return += reward
        done = terminated or truncated

        if done:
            if episode == 0:
                global_running_reward = episode_return
            else:
                global_running_reward = 0.99 * global_running_reward + 0.01 * episode_return

            episode_returns.append(episode_return)
            episodes_timesteps.append(step + 1)
            alpha = agent.log_alpha.exp().item()
            print(
                f"Ep {episode:5d} | Step {step+1:>8,} | "
                f"Return {episode_return:>9.2f} | "
                f"Running {global_running_reward:>9.2f} | "
                f"Alpha {alpha:.4f}"
            )
            episode_return = 0
            episode += 1
            observation, info = env.reset()

        # periodic deterministic evaluation — only after warmup, uses eval_env
        if step >= min_replay_size and step % eval_frequency == 0:
            agent.set_to_eval_mode()
            eval_returns = []
            for _ in range(num_eval_episodes):
                eval_obs, _ = eval_env.reset()
                eval_return = 0
                eval_done   = False
                while not eval_done:
                    obs_t       = torch.tensor(eval_obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
                    eval_action = agent.actor.act(obs_t).cpu().numpy()[0]
                    eval_obs, eval_reward, term, trunc, _ = eval_env.step(eval_action)
                    eval_return += eval_reward
                    eval_done   = term or trunc
                eval_returns.append(eval_return)
            agent.set_to_train_mode()

            mean_eval = np.mean(eval_returns)
            evaluation_returns.append(mean_eval)
            evaluation_timesteps.append(step)
            print(
                f"  >>> Eval @ step {step+1:,} | "
                f"Mean: {mean_eval:.2f} | "
                f"Running: {global_running_reward:.2f} | "
                f"Alpha: {agent.log_alpha.exp().item():.4f}"
            )

    return evaluation_returns, evaluation_timesteps, episode_returns, episodes_timesteps