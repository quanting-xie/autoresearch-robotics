"""
Fixed evaluation infrastructure for autoresearch-robotics.
Evaluation harness and rendering pipeline.

This file is READ-ONLY — the agent must NOT modify it.

Usage:
    python evaluate.py --render     # verify rendering pipeline
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

from prepare import make_env, flatten_obs, get_action_dim, ENV_ID, FRAME_WIDTH, FRAME_HEIGHT

# ---------------------------------------------------------------------------
# Early-stop helpers (DO NOT MODIFY OR REMOVE — used by train.py to cap
# wasted compute on diverging or already-converged runs)
# ---------------------------------------------------------------------------
# train.py periodically calls run_spot_eval() during training, appends the
# result to a list, and consults should_stop_training() to decide whether
# to break out of the training loop early. The TIME_BUDGET in prepare.py
# becomes a ceiling — most experiments end well before it.

import math


def run_spot_eval(policy_fn, env_id=ENV_ID, n_episodes=5):
    """Quick mid-training success-rate check. Deterministic policy, 5 episodes.
    Returns a float in [0, 1]. Cost: ~250 env steps (~3-5 sec wall clock).
    Does not write renders, does not affect the main eval that determines
    keep/discard at end of training."""
    env = make_env(env_id)
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_success = False
        while not done:
            action = policy_fn(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if float(info.get("is_success", 0.0)) > 0.5:
                ep_success = True
            done = terminated or truncated
        successes += int(ep_success)
    env.close()
    return successes / n_episodes


def should_stop_training(spot_eval_history, total_training_time, critic_loss, actor_loss,
                         min_train_seconds=600.0):
    """Decide whether to bail out of training early.
    Returns (stop: bool, reason: str). Reasons:
      - "divergence" — NaN/inf or absurd loss values (kill immediately)
      - "converged"  — last 3 spot-evals within 0.05 of each other AND mean >= 0.5
      - "no-progress" — > min_train_seconds elapsed AND last 3 spot-evals all 0.0
      - "" — keep training
    """
    # 1. Divergence: NaN/inf or extreme magnitudes. Kill at any time.
    for val in (critic_loss, actor_loss):
        if val is None:
            continue
        if math.isnan(val) or math.isinf(val):
            return True, "divergence — NaN/inf in loss"
        if abs(val) > 1e6:
            return True, f"divergence — loss magnitude {val:.2e}"

    # 2 & 3 need at least 3 spot-evals.
    if len(spot_eval_history) < 3:
        return False, ""
    last3 = spot_eval_history[-3:]

    # 2. Convergence: stable and high.
    if max(last3) - min(last3) < 0.05 and sum(last3) / 3 >= 0.5:
        return True, f"converged — last 3 spot-evals = {last3}"

    # 3. No progress: long elapsed time, still 0 across the board.
    if total_training_time > min_train_seconds and all(s == 0 for s in last3):
        return True, f"no-progress — {min_train_seconds:.0f}s elapsed, last 3 spot-evals all 0"

    return False, ""


# ---------------------------------------------------------------------------
# Protocol guard (DO NOT MODIFY OR REMOVE)
# ---------------------------------------------------------------------------
# Refuses to load if the most recent experiment in experiment_history.json
# has empty bookkeeping fields. The agent's program.md mandates that the
# SYNTHESIZE step (step 6) be completed before running another experiment.
# Without this guard the agent loop drifts under context pressure: it runs
# train.py without ever filling in hypothesis/lesson/vlm fields, so each
# subsequent ANALYZE has no real memory and the loop becomes random search.

def _check_prior_bookkeeping_complete():
    p = Path("./experiment_history.json")
    if not p.exists():
        return  # first experiment; nothing to check
    try:
        h = json.loads(p.read_text())
    except Exception:
        return  # malformed json; let the rest of the script handle it
    exps = h.get("experiments", []) or []
    if not exps:
        return
    last = exps[-1]
    required = ("hypothesis", "vlm_feedback_summary", "status", "lesson_learned")
    missing = [k for k in required if not (last.get(k) or "").strip()]
    if not missing:
        return
    raise RuntimeError(
        f"\n"
        f"────────────────────────────────────────────────────────────\n"
        f"PROTOCOL VIOLATION — bookkeeping incomplete for experiment {last.get('id')}.\n"
        f"\n"
        f"Missing fields: {missing}\n"
        f"\n"
        f"Per program.md step 6 (SYNTHESIZE), every experiment must have\n"
        f"hypothesis / vlm_feedback_summary / status / lesson_learned filled\n"
        f"before another experiment can run. The training script will not\n"
        f"start until you backfill the latest entry.\n"
        f"\n"
        f"To fix:\n"
        f"  1. Read renders_archive/exp{last.get('id')}/*.png for visual context\n"
        f"     (or `git show {last.get('commit') or 'HEAD'}` if archive is missing)\n"
        f"  2. Update the empty fields in experiment_history.json\n"
        f"  3. Commit experiment_history.json (`git add experiment_history.json && git commit`)\n"
        f"  4. Re-run `uv run train.py --headless`\n"
        f"────────────────────────────────────────────────────────────"
    )

_check_prior_bookkeeping_complete()

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

EVAL_EPISODES = 10           # episodes for quantitative evaluation
RENDER_EPISODES = 3          # episodes to render for visual analysis

# ---------------------------------------------------------------------------
# Evaluation harness (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(policy_fn, env_id=ENV_ID, n_episodes=EVAL_EPISODES):
    """Run evaluation episodes and compute metrics.

    Args:
        policy_fn: callable(obs_dict) -> action (np.ndarray)
            Takes a raw observation dict and returns an action.
        env_id: environment ID
        n_episodes: number of evaluation episodes

    Returns:
        dict with keys:
            success_rate: fraction of episodes achieving the goal
            mean_reward: average episode return
            mean_distance: average final distance to goal
            per_episode: list of per-episode dicts
    """
    env = make_env(env_id)
    results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        # Compute final distance to goal
        final_distance = np.linalg.norm(
            obs["achieved_goal"] - obs["desired_goal"]
        )
        success = info.get("is_success", float(final_distance < 0.05))

        results.append({
            "reward": episode_reward,
            "distance": final_distance,
            "success": float(success),
        })

    env.close()

    success_rate = np.mean([r["success"] for r in results])
    mean_reward = np.mean([r["reward"] for r in results])
    mean_distance = np.mean([r["distance"] for r in results])

    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "mean_distance": mean_distance,
        "per_episode": results,
    }

# ---------------------------------------------------------------------------
# Rendering pipeline (headless MuJoCo)
# ---------------------------------------------------------------------------

def render_episodes(policy_fn, env_id=ENV_ID, n_episodes=RENDER_EPISODES,
                    output_dir="./renders", show_window=False):
    """Render evaluation episodes with MuJoCo.

    Captures frames at each timestep, saves a video and key frame PNGs.
    Optionally displays a live MuJoCo window.

    Args:
        policy_fn: callable(obs_dict) -> action
        env_id: environment ID
        n_episodes: number of episodes to render
        output_dir: directory to save renders
        show_window: if True, replay episodes in a live MuJoCo window after recording

    Returns:
        dict with paths to saved files
    """
    import imageio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(env_id, render_mode="rgb_array")
    all_frames = []
    episode_boundaries = [0]

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        # Capture initial frame
        frame = env.render()
        all_frames.append(frame)

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            all_frames.append(frame)
            done = terminated or truncated

        episode_boundaries.append(len(all_frames))

    env.close()

    # Save video
    video_path = output_dir / "eval_video.mp4"
    imageio.mimsave(str(video_path), all_frames, fps=30)

    # Save key frames: first, middle, last of each episode
    frame_paths = []
    for ep in range(n_episodes):
        start = episode_boundaries[ep]
        end = episode_boundaries[ep + 1]
        mid = (start + end) // 2

        for label, idx in [("start", start), ("mid", mid), ("end", end - 1)]:
            fname = f"ep{ep}_{label}.png"
            fpath = output_dir / fname
            imageio.imwrite(str(fpath), all_frames[idx])
            frame_paths.append(str(fpath))

    # Live window display (optional)
    if show_window:
        print("Opening live MuJoCo window...")
        # Switch to GLFW backend for window display.
        # EGL (used for offscreen above) can't open windows.
        # WSLg provides a local X11 display at :0.
        prev_gl = os.environ.get("MUJOCO_GL")
        prev_display = os.environ.get("DISPLAY")
        os.environ["MUJOCO_GL"] = "glfw"
        os.environ["DISPLAY"] = ":0"

        env_human = make_env(env_id, render_mode="human")
        for ep in range(n_episodes):
            obs, info = env_human.reset()
            done = False
            while not done:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, info = env_human.step(action)
                done = terminated or truncated
        env_human.close()

        # Restore previous backend
        if prev_gl is not None:
            os.environ["MUJOCO_GL"] = prev_gl
        else:
            os.environ.pop("MUJOCO_GL", None)
        if prev_display is not None:
            os.environ["DISPLAY"] = prev_display
        else:
            os.environ.pop("DISPLAY", None)

    return {
        "video_path": str(video_path),
        "frame_paths": frame_paths,
        "n_frames": len(all_frames),
    }

# ---------------------------------------------------------------------------
# Main (verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify evaluation infrastructure")
    parser.add_argument("--render", action="store_true", help="Test rendering pipeline")
    args = parser.parse_args()

    # Random policy for testing
    action_dim = get_action_dim(ENV_ID)
    def random_policy(obs):
        return np.random.uniform(-1, 1, size=action_dim)

    # Evaluate with random policy
    print("Running evaluation with random policy...")
    metrics = evaluate(random_policy, ENV_ID, n_episodes=5)
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  Mean reward: {metrics['mean_reward']:.2f}")
    print(f"  Mean distance: {metrics['mean_distance']:.4f}")
    print()

    if args.render:
        print("Testing rendering pipeline...")
        render_result = render_episodes(random_policy, ENV_ID, n_episodes=2, output_dir="./renders")
        print(f"  Video: {render_result['video_path']}")
        print(f"  Frames: {len(render_result['frame_paths'])} key frames saved")
        print(f"  Total frames captured: {render_result['n_frames']}")
        print()

    print("Done!")
