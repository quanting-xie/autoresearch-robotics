"""
Prototype harness for LIBERO (robosuite, image+proprio observations).

This file is a STUB. setup_task.py recognizes it as a prototype because of
the NotImplementedError below. To turn this into a real template, you need to:

  1. Add `libero` and `robosuite` to a per-template pyproject.toml override.
  2. Replace make_env() with `libero.libero.benchmark.make_env(task_idx=...)`
     or your own factory returning a robosuite env.
  3. Replace flatten_obs() with logic that combines RGB (`agentview_image`)
     with proprioceptive vectors (`robot0_eef_pos`, `robot0_eef_quat`,
     `robot0_gripper_qpos`) — typically a CNN encoder feeding a vector head.
  4. Add a per-template evaluate.py with the LIBERO success metric (terminal
     reward = 1 on success, 0 otherwise — accumulate per episode).
  5. The rendering pipeline can capture `agentview_image` directly each step;
     you don't need MuJoCo's offscreen renderer.

Action space: 7-D float in [-1, 1] — robosuite OSC_POSE
    [dx, dy, dz, drx, dry, drz, gripper(+1=close,-1=open)]

See /home/quanting/Project/Auto-research/Auto-research/libero_pro_eval.py
in the parent project for a working LIBERO evaluator that you can crib from.
"""

import numpy as np

TIME_BUDGET = 600            # 10 minutes — LIBERO needs more than Fetch
MAX_EPISODE_STEPS = 200
FRAME_WIDTH = 256
FRAME_HEIGHT = 256
ENV_ID = "libero/lift"


def make_env(env_id=ENV_ID, render_mode=None):
    raise NotImplementedError(
        "LIBERO harness is a prototype stub. See module docstring for the "
        "5-step recipe to implement it."
    )


def flatten_obs(obs_dict):
    raise NotImplementedError(
        "LIBERO obs combines RGB (agentview_image) and proprio vectors. "
        "Implement a CNN+MLP encoder or use an existing VLA backbone."
    )


def get_obs_dim(env_id=ENV_ID):
    raise NotImplementedError("Depends on encoder design.")


def get_action_dim(env_id=ENV_ID):
    return 7  # robosuite OSC_POSE


def get_action_bounds(env_id=ENV_ID):
    return (np.full(7, -1.0, dtype=np.float32),
            np.full(7,  1.0, dtype=np.float32))


if __name__ == "__main__":
    print(f"LIBERO harness — prototype stub for {ENV_ID}")
    print("This template is not yet implemented. See the module docstring.")
