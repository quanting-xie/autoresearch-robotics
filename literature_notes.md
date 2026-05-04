# Literature notes — autoresearch/may3

Distilled from a web scan on 2026-05-03 before experiment 1. Re-read this on
exp 2+. Re-search only when starting a genuinely new direction or after ≥3
stalled experiments.

## Constraint reminder

- Single env (no parallel collection); 30 min wall-clock budget per experiment.
- No new packages — torch/gymnasium/gymnasium-robotics/mujoco/numpy/imageio/wandb only.
- Env is `FetchPickAndPlace-v4`; sparse reward; obs dim 28 (incl. desired_goal); action dim 4.
- Baseline reaches ~170 steps/sec on RTX 5090 → ~300K env steps in 30 min.
- OpenAI's HER baseline (Andrychowicz 2017) used 19 MPI processes — not comparable.
  With single-env + 30 min, realistic targets are much lower than the 90%+ in their paper.

## Algorithmic ideas worth trying

### CrossQ (Bhatt et al., 2024) — arxiv 1902.05605v4
- **Idea (one sentence):** Replace target networks with BatchRenorm, use a wider critic (2048 hidden), single forward pass with concatenated current+next state-action pairs to keep BN statistics consistent.
- **Reported:** Matches REDQ/DroQ at UTD=1 (≈4× wallclock speedup). State-of-the-art sample efficiency on MuJoCo continuous control.
- **Applicable here?** YES — pure architectural change, no extra deps. NOT yet tested with sparse rewards / HER, so it's a research bet, not a sure thing. Dropping target nets simplifies code.
- **Risk:** wider critic (2048) increases VRAM; need to check fit. Could try a smaller width first.

### Spectral / layer normalization in actor-critic
- **Idea:** Normalize layers in critic (and sometimes actor) for stability, especially with deeper networks. Bjorck et al. (NeurIPS 2021) "Towards Deeper Deep RL with Spectral Normalization" showed deeper networks become trainable.
- **Applicable here?** YES — trivial PyTorch change (`nn.LayerNorm` or `spectral_norm`). Cheap to try. Often adds 5-10% stability on its own.

### MHER / Multi-step HER (Yang et al., arxiv 2102.12962)
- **Idea:** Use n-step returns inside HER. Naive n-step + HER introduces off-policy bias; the paper proposes MHER(λ) and Model-based MHER to correct it.
- **Reported:** Significantly higher sample efficiency than vanilla HER on Fetch-style multi-goal tasks.
- **Applicable here?** PARTIAL — MHER(λ) is implementable in pure PyTorch, no new deps; MMHER needs a learned dynamics model (more code). Worth trying MHER(λ) as a relatively contained change.
- **Risk:** more complex sampling code; bias correction is fiddly.

### GCHR — Goal-Conditioned Hindsight Regularization (arxiv 2508.06108)
- **Idea:** Hindsight goal-conditioned action regularization (HGR) + hindsight self-imitation (HSR) on top of HER.
- **Reported:** Substantially better sample reuse on navigation + manipulation than HER + self-imitation baselines.
- **Applicable here?** Maybe — abstract is thin; would need to implement regularization + self-imitation losses on top of SAC. Mid-complexity change.

### FAHER — Failed-goal-Aware HER (PeerJ 2024)
- **Idea:** Cluster episodes in the buffer by failed goals; bias HER sampling toward useful clusters.
- **Couldn't fetch full text (403).** Treat as background context only.

## Sanity checks from older work

- **DDPG+HER baseline performance on FetchPickAndPlace** (with demos: ~100% by epoch 400; without demos: ~70% after 1000 epochs). Our compute is ≪ that — first-experiment success rate could realistically be 0–30%.
- **HER strategy:** "future" with k≈4 is the standard. "final" is occasionally competitive on short episodes (50 steps here).
- **TD3 vs SAC:** TD3 trains more stably but tends to plateau lower on harder tasks; SAC's entropy regularization helps exploration in sparse-reward settings. Sticking with SAC family is reasonable; TD3 swap is a backup direction.

## Priority queue for experiments 2+

Ordered by (expected gain × ease of implementation × novelty):

1. **LayerNorm in actor + critic** — cheapest, well-supported by literature; should help stability.
2. **More gradient updates per env step** (UTD ratio > 1, e.g. 4) — classic sample-efficiency lever; SAC variants like REDQ rely on this. Caveat: slows wallclock.
3. **CrossQ-style: drop target net + BatchRenorm** — bigger structural change but documented gains. Uncertain interaction with HER.
4. **MHER(λ)** — non-trivial implementation but directly attacks HER's main weakness on this exact task family.
5. **HER k tuning** (k=4 → k=8 or k=2) — cheap ablation; only worth it if other levers exhaust.
6. **Switch to TD3** — backup direction; only if SAC keeps thrashing.

## Failed direction notes (carry forward)

(none yet — to be populated as experiments fail)
