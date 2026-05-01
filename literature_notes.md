# Literature notes — autoresearch/may1

Distilled findings from web searches. Every entry assessed for applicability under our constraints: 30-min wall-clock budget, fixed `FetchPickAndPlace-v4` env, no new dependencies (only torch / numpy / gymnasium / mujoco / imageio / Pillow are installed).

## Most directly applicable

### "Efficient Sparse-Reward Goal-Conditioned RL with High Replay Ratio and Regularization" (arXiv 2312.05787)
- Target domain matches ours exactly: sparse-reward goal-conditioned Fetch tasks. Reports ~8× sample efficiency on Fetch tasks vs SoTA SAC+HER.
- Recipe (REDQ+HER+BQ):
  - **Replay ratio G = 20** (20 gradient updates per env step)
  - **Ensemble of N = 10** Q-functions
  - **LayerNorm after every weight layer in critic** — critical for stable high-RR training.
  - **Q-value bounding**: target y = r + γ·clamp( min over subset of Q̄ − α·log π , Q_min, Q_max ), with Q_max = 0 and Q_min = -1/(1-γ). For γ=0.98 → Q_min = -50. This is huge for sparse rewards.
  - REMOVED from REDQ: clipped double-Q (replaced by ensemble mean), entropy term in target.
- **Applicability**: directly relevant. The 30-min budget is wall-clock not steps, so RR=20 is risky throughput-wise. But LayerNorm + Q-clipping in particular are nearly free to add and target the exact bottleneck (sparse reward, slow value propagation).

### BRO — "Bigger, Regularized, Optimistic" (arXiv 2405.16158)
- 400% improvement over SAC at comparable wall clock (BRO Fast).
- Recipe: critic = ResNet blocks of (Dense→LayerNorm→ReLU), width 512, ~5M params. Replay ratio 10. Weight decay. Periodic full network resets. Optimistic dual-actor exploration (KL-regularized Q upper bound from quantile ensemble).
- **Applicability**: partial. The architecture (LayerNorm everywhere + residual blocks) is portable to PyTorch. Optimistic dual-actor and quantile ensemble are heavier and less obviously needed for FetchPickAndPlace. JAX-only reference impl, but the recipe is just torch nn modules. **Best candidate for "structural" experiments after we exhaust simpler levers.**

### Maximum Entropy HER (arXiv 2410.24016)
- Selectively applies HER. Originally for PPO. Goal-sampling tweak only.
- **Applicability**: low — abstract is light on details, gains were in on-policy / Predator-Prey. Skip for now.

## General takeaways from advances-rl-sota-2026 summary
- LayerNorm in critics is now considered baseline practice; "naive" SAC without it is suboptimal.
- Higher replay ratios work IF networks are regularized (LayerNorm, weight decay, periodic resets).
- CrossQ (BatchNorm in critics) gives REDQ-level sample efficiency without the ensemble — but requires careful BN handling. Riskier than LayerNorm.

## Action plan based on literature
1. **Baseline first** (mandated). Establish number.
2. **First intervention candidates** (cheap, high-confidence):
   - Add LayerNorm to critic (and maybe actor) — confirmed staple of every recent paper.
   - Add Q-value bounding (clamp target Q to [-1/(1-γ), 0]) — uniquely targets sparse-reward Fetch.
3. **Second wave** (if 1+2 land):
   - Increase replay ratio (UPDATE_EVERY decrease or N_UPDATES increase) — only valuable once LayerNorm is in.
4. **Radical** (if plateau):
   - BRO-style ResNet critic with width 512.
   - Periodic critic reset.
   - Quantile / distributional Q (more code, marginal expected gain on this env).

## Constraint reminder
- 30 min training wall clock is the budget. Higher RR multiplies grad-step cost. Need to measure steps/sec impact before committing to RR=20.
- VRAM soft constraint — RTX 5090 has 32GB free, plenty of room for bigger nets/ensembles.
- Cannot install new packages (no JAX, no D4RL).
