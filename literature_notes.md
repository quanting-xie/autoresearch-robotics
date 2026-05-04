# Literature notes — autoresearch/reach-may3

Distilled findings relevant to SAC+HER on FetchReach-v4 with a **10-second wall-clock training budget** (the dominant constraint — most papers assume orders of magnitude more compute).

## Sample-efficient SAC family

- **REDQ — Randomized Ensembled Double Q-Learning** (Chen et al., arxiv 2101.05982). Idea: raise update-to-data (UTD) ratio from 1 → 20, but use a 10-network Q ensemble with random subsample-of-2 per target to control bias. Matches model-based sample efficiency on MuJoCo. *Applicable here:* high UTD is the single most relevant trick — our wall-clock budget allows a fixed number of env steps but plenty of GPU cycles for extra gradient updates. Ensemble cost is acceptable on tiny FetchReach nets. No new dep.
- **DroQ — Dropout-Q** (Hiraoka et al., 2110.02034). Same idea as REDQ but uses dropout + LayerNorm on the critic instead of an ensemble — cheaper. *Applicable here:* yes, even simpler. No new dep.
- **CrossQ** (Bhatt et al., 1902.05605 — recently revived). Removes target networks entirely, uses BatchNorm on the critic, matches REDQ sample efficiency at 1× UTD. *Applicable here:* removing target nets is appealing for simplicity, but BN behavior under tiny batches and short runs is fragile. Hold for later if simpler tricks aren't enough.
- **SAC Flow** (arxiv 2509.25756, late 2025). Flow-based policy via velocity-reparameterized sequential modeling. *Not applicable* — requires building a flow model, way too heavyweight for a 10-second budget on a toy task.

## Goal-conditioned RL

- **HER** (Andrychowicz et al., 1707.01495) — already in baseline, k=4, future strategy. Standard.
- **GCPO — Goal-conditioned on-policy RL** (NeurIPS 2024). On-policy variant. *Not applicable* — on-policy is sample-inefficient under tight budgets; off-policy + HER beats it on FetchReach.
- **Dense reward shaping** for goal-conditioned tasks (multiple 2024-2025 surveys). Replacing sparse {-1, 0} with `-||achieved - desired||` (or potential-based shaping) typically halves the steps to convergence on FetchReach because the gradient is well-defined every step. *Applicable here:* high-priority candidate for early experiments. The eval metric is fixed (it counts successes), so we can shape *training* reward freely.

## Other ideas filed for later

- **n-step returns** (n=3-5) on the critic: known to speed convergence on dense-reward tasks. Cheap to implement. No new dep.
- **Larger batch / more grad steps per env step** (UTD ↑) is the lowest-hanging fruit if STEPS_BEFORE_LEARNING is reduced first.
- **STEPS_BEFORE_LEARNING=1000** in a 10s budget is suspect — at FetchReach's typical step rate this can be a sizeable fraction of the run before the first gradient step. Lowering to 100-200 is an obvious early experiment.
- **Smaller networks**: FetchReach has only 13-dim obs and 4-dim actions. The default 256×3 may be over-parameterized; a 128×2 or 64×2 net might train faster wall-clock.

## Key numbers

- Env: FetchReach-v4, max_episode_steps=50, sparse reward in env (-1 per step until distance < 0.05).
- Eval: 10 episodes deterministic; success = `is_success` from info (also distance < 0.05).
- Action range: [-1, 1], dim 4.
- Obs (flat): 13 (10 obs + 3 desired_goal).

## Constraints reminder

- Cannot install new packages. Available: torch, gymnasium, gymnasium-robotics, mujoco, numpy, imageio, Pillow, wandb. (PyTorch has Dropout, LayerNorm, BatchNorm built-in — all REDQ/DroQ/CrossQ tricks are implementable.)
- Cannot modify `prepare.py` or `evaluate.py`. The eval reward is sparse from the env, but the *training* reward (in HER relabeling and via reward shaping in `train.py`) is fully under our control.
