"""
go1_direction_env.py — Minimal reward, maximum learning signal.

Core insight: the robot should only get reward for actually moving forward.
Sitting still = 0 reward. Moving forward = positive reward. Falling = episode ends.

Reward:
  r = r_forward + r_heading - r_torque - termination_penalty
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

MJCF_PATH = os.environ.get("GO1_MJCF_PATH", "unitree_go1/scene.xml")
GO1_STANDING_HEIGHT = 0.27


class Go1DirectionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    DT           = 0.002
    FRAME_SKIP   = 10
    MAX_STEPS    = 1000
    Z_NOMINAL    = GO1_STANDING_HEIGHT
    Z_MIN        = 0.15
    ACTION_ALPHA = 0.7

    # ── Reward ────────────────────────────────────────────────────────────────
    # r_forward: raw forward velocity — no exp, no target speed
    #   positive when moving in commanded direction
    #   negative when moving away from it
    #   sitting still = exactly 0
    W_FORWARD     = 2.0

    # r_heading: cosine of angle between body forward and commanded direction
    #   1.0 when perfectly aligned, -1 when facing backwards
    #   encourages robot to face where it walks
    W_HEADING     = 0.5

    # r_torque: small penalty so robot doesn't thrash joints unnecessarily
    W_TORQUE      = 0.002

    # termination penalty: large one-time penalty for falling
    W_TERMINATION = 5.0
    W_LATERAL     = 1.0    # penalize sideways drift

    DIRECTION_2D = False

    def __init__(
        self,
        mjcf_path       = MJCF_PATH,
        render_mode     = None,
        fixed_direction = None,
        direction_2d    = None,
        seed            = None,
    ):
        super().__init__()
        self.render_mode     = render_mode
        self.fixed_direction = fixed_direction
        self.direction_2d    = direction_2d if direction_2d is not None else self.DIRECTION_2D
        self._step_count     = 0
        self._renderer       = None

        import mujoco
        self._mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._mj_data  = mujoco.MjData(self._mj_model)
        self._mj_model.opt.timestep = self.DT
        self._nu = self._mj_model.nu  # 12

        # Observation:
        # joint_pos(12) + joint_vel(12) + base_linvel(3) + base_angvel(3)
        # + gravity_proj(3) + prev_action(12) + direction(3) = 48
        obs_dim = 12 + 12 + 3 + 3 + 3 + 12 + 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._nu,), dtype=np.float32
        )

        self._direction   = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._prev_action = np.zeros(self._nu, dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count  = 0
        self._prev_action = np.zeros(self._nu, dtype=np.float32)

        if self.fixed_direction is not None:
            d = np.asarray(self.fixed_direction, dtype=np.float32)
        else:
            d = self._sample_direction()
        self._direction = d / (np.linalg.norm(d) + 1e-8)

        import mujoco
        mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, 0)
        self._mj_data.qpos[7:] += self.np_random.uniform(
            -0.02, 0.02, size=self._mj_model.nq - 7
        )
        mujoco.mj_forward(self._mj_model, self._mj_data)

        return self._get_obs(), {"direction": self._direction.copy()}

    def step(self, action):
        import mujoco
        self._step_count += 1

        action = np.clip(action, -1.0, 1.0)
        smooth = self.ACTION_ALPHA * self._prev_action + (1 - self.ACTION_ALPHA) * action

        ctrl_min = self._mj_model.actuator_ctrlrange[:, 0]
        ctrl_max = self._mj_model.actuator_ctrlrange[:, 1]
        self._mj_data.ctrl[:] = 0.5 * (smooth + 1.0) * (ctrl_max - ctrl_min) + ctrl_min

        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self._mj_model, self._mj_data)

        z_height  = float(self._mj_data.qpos[2])
        roll, pitch = self._get_roll_pitch()

        terminated = bool(
            z_height < self.Z_MIN or
            abs(roll)  > np.deg2rad(45) or
            abs(pitch) > np.deg2rad(45)
        )
        truncated = bool(self._step_count >= self.MAX_STEPS)

        reward, info = self._compute_reward(terminated)

        self._prev_action = smooth.copy()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            import mujoco
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self._mj_model, height=480, width=640
                )
            try:
                self._renderer.update_scene(self._mj_data, camera="tracking")
            except Exception:
                self._renderer.update_scene(self._mj_data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self):
        d = self._mj_data
        qpos         = d.qpos[7:].astype(np.float32)           # 12
        qvel         = d.qvel[6:].astype(np.float32)           # 12
        base_linvel  = d.qvel[:3].astype(np.float32)           # 3  world frame
        base_angvel  = d.qvel[3:6].astype(np.float32)          # 3
        base_xmat    = d.xmat[1].reshape(3, 3)
        gravity_base = (base_xmat.T @ np.array([0., 0., -1.])).astype(np.float32)  # 3
        prev_action  = self._prev_action.astype(np.float32)    # 12
        direction    = self._direction.astype(np.float32)      # 3

        return np.concatenate([
            qpos, qvel, base_linvel, base_angvel,
            gravity_base, prev_action, direction
        ])

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_roll_pitch(self):
        w, x, y, z = self._mj_data.qpos[3:7]
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        return float(roll), float(pitch)

    def _sample_direction(self):
        if self.direction_2d:
            angle = self.np_random.uniform(0, 2 * np.pi)
            return np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        phi   = self.np_random.uniform(0, 2 * np.pi)
        theta = self.np_random.uniform(0, np.pi / 3)
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ], dtype=np.float32)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, terminated):
        d        = self._mj_data
        base_vel = d.qvel[:3].copy()   # world frame linear velocity

        # 1. Forward velocity along commanded direction
        vel_along  = float(np.dot(base_vel, self._direction))

        # 2. Heading — cosine between body forward axis and commanded direction
        #    1.0 = facing exactly right, 0.0 = facing 90deg sideways, -1 = backwards
        base_xmat    = d.xmat[1].reshape(3, 3)
        body_forward = base_xmat[:, 0]   # robot's forward axis in world frame
        heading_cos  = float(np.dot(body_forward, self._direction))
        heading_cos  = max(heading_cos, 0.0)  # no reward for facing backwards

        # Combined: velocity reward GATED by heading alignment
        # If heading_cos = 1.0 (perfectly aligned) → full velocity reward
        # If heading_cos = 0.5 (45 deg sideways)   → half velocity reward
        # If heading_cos = 0.0 (90 deg sideways)   → zero velocity reward
        # This forces the robot to FACE the direction it moves in
        r_forward  = self.W_FORWARD * vel_along * heading_cos

        # Heading bonus — small additional reward for good alignment
        # Keeps heading signal alive even when standing still
        r_heading  = self.W_HEADING * heading_cos

        # 3. Lateral velocity penalty — penalize moving sideways
        lateral_vel   = base_vel - vel_along * self._direction
        lateral_speed = float(np.linalg.norm(lateral_vel))
        r_lateral     = -self.W_LATERAL * lateral_speed

        # 4. Torque penalty — tiny, just prevents extreme joint forces
        torques   = d.actuator_force
        r_torque  = -self.W_TORQUE * float(np.mean(torques ** 2))

        # 5. Termination penalty
        r_term    = -self.W_TERMINATION if terminated else 0.0

        reward = r_forward + r_lateral + r_heading + r_torque + r_term

        info = {
            "vel_along":   vel_along,
            "heading_cos": heading_cos,
            "r_forward":   r_forward,
            "r_lateral":   r_lateral,
            "lateral_speed": lateral_speed,
            "r_heading":   r_heading,
            "r_torque":    r_torque,
            "r_term":      r_term,
            "z_height":    float(d.qpos[2]),
            "direction":   self._direction.copy(),
        }
        return reward, info


gym.register(
    id="Go1Direction-v0",
    entry_point=f"{__name__}:Go1DirectionEnv",
    max_episode_steps=1000,
)