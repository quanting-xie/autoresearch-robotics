"""
Fixed evaluation infrastructure for autoresearch-robotics — Pendulum / classic control.
Evaluation harness and rendering pipeline.

This file is READ-ONLY — the agent must NOT modify it.

Differs from core/evaluate.py in the success metric:
  - core: success = ||achieved_goal - desired_goal|| < 0.05  (goal-conditioned)
  - here: success = episode_return >= SUCCESS_THRESHOLD       (return-threshold)

The returned dict shape is identical so train.py works unchanged.
`mean_distance` is repurposed as `mean(|final_obs[2]|)` (final |theta_dot|),
which is a meaningful "how settled is the pendulum at episode end" signal.
"""

import os
from pathlib import Path

import numpy as np

from prepare import make_env, flatten_obs, get_action_dim, ENV_ID, SUCCESS_THRESHOLD

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

    Returns dict with:
        success_rate:  fraction of episodes with return >= SUCCESS_THRESHOLD
        mean_reward:   average episode return
        mean_distance: average |final omega| (Pendulum-specific "settledness")
        per_episode:   list of per-episode dicts
    """
    env = make_env(env_id)
    results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        last_obs_vec = flatten_obs(obs)
        done = False

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            last_obs_vec = flatten_obs(obs)
            done = terminated or truncated

        # Pendulum obs = [cos(theta), sin(theta), theta_dot]; |theta_dot| at end
        # is a settledness proxy. For non-Pendulum classic control envs you'd
        # adapt this — but the contract (return one scalar "auxiliary metric"
        # under the key `mean_distance`) is preserved.
        if last_obs_vec.shape[0] >= 3:
            settledness = float(abs(last_obs_vec[2]))
        else:
            settledness = float(np.linalg.norm(last_obs_vec))

        success = float(episode_reward >= SUCCESS_THRESHOLD)

        results.append({
            "reward": episode_reward,
            "distance": settledness,
            "success": success,
        })

    env.close()

    success_rate = float(np.mean([r["success"] for r in results]))
    mean_reward = float(np.mean([r["reward"] for r in results]))
    mean_distance = float(np.mean([r["distance"] for r in results]))

    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "mean_distance": mean_distance,
        "per_episode": results,
    }

# ---------------------------------------------------------------------------
# Rendering pipeline
# ---------------------------------------------------------------------------

def render_episodes(policy_fn, env_id=ENV_ID, n_episodes=RENDER_EPISODES,
                    output_dir="./renders", show_window=False):
    """Render evaluation episodes. Same shape as core/evaluate.py."""
    import imageio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(env_id, render_mode="rgb_array")
    all_frames = []
    episode_boundaries = [0]

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

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

    video_path = output_dir / "eval_video.mp4"
    imageio.mimsave(str(video_path), all_frames, fps=30)

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

    if show_window:
        print("Opening live render window...")
        env_human = make_env(env_id, render_mode="human")
        for ep in range(n_episodes):
            obs, info = env_human.reset()
            done = False
            while not done:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, info = env_human.step(action)
                done = terminated or truncated
        env_human.close()

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

    action_dim = get_action_dim(ENV_ID)
    def random_policy(obs):
        return np.random.uniform(-2, 2, size=action_dim)

    print("Running evaluation with random policy...")
    metrics = evaluate(random_policy, ENV_ID, n_episodes=5)
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  Mean reward: {metrics['mean_reward']:.2f}")
    print(f"  Mean |omega| at end: {metrics['mean_distance']:.4f}")
    print()

    if args.render:
        print("Testing rendering pipeline...")
        render_result = render_episodes(random_policy, ENV_ID, n_episodes=2, output_dir="./renders")
        print(f"  Video: {render_result['video_path']}")
        print(f"  Frames: {len(render_result['frame_paths'])} key frames saved")
        print(f"  Total frames captured: {render_result['n_frames']}")
        print()

    print("Done!")
