"""
Fixed infrastructure for autoresearch-robotics — Pendulum-v1 (classic control).
Environment factory and observation utilities.

This file is READ-ONLY — the agent must NOT modify it.

This template demonstrates the *non-goal-conditioned* harness contract.
The env is plain Gymnasium (no `achieved_goal` / `desired_goal` keys).
A small shim wraps the raw obs into a dict so the SAC+HER baseline in
core/train.py imports cleanly, but HER will not produce useful relabeled
rewards on this env — your first experiment should rewrite train.py to
drop HER and use plain SAC (or your algorithm of choice).

Usage:
    python prepare.py              # verify environment works
"""

import os

import numpy as np


# ---------------------------------------------------------------------------
# Auto-configure MuJoCo OpenGL backend (Pendulum's render uses pygame, not
# MuJoCo, so this is harmless but kept for parity with other templates)
# ---------------------------------------------------------------------------

def _configure_mujoco_gl():
    if "MUJOCO_GL" in os.environ:
        return
    try:
        import ctypes
        ctypes.cdll.LoadLibrary("libEGL.so.1")
        os.environ["MUJOCO_GL"] = "egl"
    except OSError:
        os.environ["MUJOCO_GL"] = "osmesa"

_configure_mujoco_gl()

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 60             # training time budget in seconds (1 minute)
MAX_EPISODE_STEPS = 200      # Pendulum-v1 default
FRAME_WIDTH = 500            # Pendulum default render size
FRAME_HEIGHT = 500
ENV_ID = "Pendulum-v1"

# Success threshold: episode return >= this counts as a success.
# Pendulum's per-step reward is in [-16.27, 0]; a perfect upright hold
# would give ~0 over an episode. -200 is a "mostly upright" episode.
SUCCESS_THRESHOLD = -200.0

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

class _GoalDictShim:
    """Wraps a Gym env to emit a dict obs of shape:
        {"observation": <vector>, "achieved_goal": [0], "desired_goal": [0]}
    so the goal-conditioned baseline in core/train.py imports cleanly.
    The zero "goals" make HER a structural no-op — the agent's first job
    is to rewrite train.py without HER.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._wrap(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._wrap(obs), reward, terminated, truncated, info

    def render(self, *a, **kw):
        return self.env.render(*a, **kw)

    def close(self):
        return self.env.close()

    @staticmethod
    def _wrap(obs):
        return {
            "observation": np.asarray(obs, dtype=np.float32),
            "achieved_goal": np.zeros(1, dtype=np.float32),
            "desired_goal": np.zeros(1, dtype=np.float32),
        }


def make_env(env_id=ENV_ID, render_mode=None):
    """Create a classic-control Gym env wrapped to emit goal-dict obs."""
    import gymnasium

    kwargs = {"max_episode_steps": MAX_EPISODE_STEPS}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gymnasium.make(env_id, **kwargs)
    return _GoalDictShim(env)


def flatten_obs(obs_dict):
    """For non-goal envs we use the raw observation only — no goal concat.

    Args:
        obs_dict: dict with "observation" key (the shim guarantees this).

    Returns:
        np.ndarray: observation vector.
    """
    return np.asarray(obs_dict["observation"], dtype=np.float32)


def get_obs_dim(env_id=ENV_ID):
    env = make_env(env_id)
    obs, _ = env.reset()
    dim = flatten_obs(obs).shape[0]
    env.close()
    return dim


def get_action_dim(env_id=ENV_ID):
    env = make_env(env_id)
    dim = env.action_space.shape[0]
    env.close()
    return dim


def get_action_bounds(env_id=ENV_ID):
    env = make_env(env_id)
    low = env.action_space.low.copy()
    high = env.action_space.high.copy()
    env.close()
    return low, high


# ---------------------------------------------------------------------------
# Main (verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Environment: {ENV_ID}")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Success threshold (episode return): {SUCCESS_THRESHOLD}")
    print()

    print("Verifying environment...")
    env = make_env(ENV_ID)
    obs, info = env.reset()
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Observation shape: {obs['observation'].shape}")
    print(f"  Flattened obs dim: {flatten_obs(obs).shape[0]}")
    print(f"  Action dim: {env.action_space.shape[0]}")
    print(f"  Action range: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    env.close()
    print("  Environment OK!")
    print()

    print("Done! Environment verified.")
