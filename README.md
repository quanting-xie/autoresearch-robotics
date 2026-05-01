# autoresearch-robotics

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) adapted for robotics -- autonomous overnight experiment optimization with robotics simulation feedback loop.

![teaser](assets/teaser.gif)

## How it works

The repo is deliberately kept small and only really has a few files that matter:

- **`prepare.py`** -- fixed constants, environment factory, observation utilities. Not modified.
- **`evaluate.py`** -- fixed evaluation harness, rendering pipeline. Not modified.
- **`train.py`** -- the single file the agent edits. Contains the full SAC+HER policy, optimizer, and training loop. Everything is fair game: architecture, hyperparameters, RL algorithm, buffer design, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** -- baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed time budget** (wall clock, excluding eval overhead), regardless of the details of your compute. The metric is **eval_success_rate** -- higher is better. Unlike standard LLM training research where the only feedback is a loss curve, robotics has a visual component: the agent analyzes MuJoCo renders of the robot's behavior alongside quantitative metrics. "The arm overshoots and oscillates" is the kind of insight that numbers alone can't give you.

## Quick start

**Requirements:** Python 3.10+, an NVIDIA GPU, [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (or any coding agent).

```bash
git clone https://github.com/jellyheadandrew/autoresearch-robotics.git
cd autoresearch-robotics

# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and verify
uv sync
uv run prepare.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

### Running the agent

Simply spin up Claude Code (and disable all permissions), then prompt:

```bash
claude --dangerously-skip-permissions
```

```
Hi! Read program.md and let's kick off a new experiment. Start the experiment loop.
# Or, for headless mode (recommended for overnight runs):
Hi! Read program.md and let's kick off a new experiment. Start the experiment loop. Use --headless mode.
```

The agent reads `program.md`, creates a branch, runs the baseline, then loops: analyze → hypothesize → modify `train.py` → commit → train → evaluate (render + visual analysis) → keep/discard → repeat. It runs indefinitely until you Ctrl+C. For overnight runs, use `tmux`.

Monitor from another terminal:

```bash
watch -n 60 'cat results.tsv'
watch -n 30 'git log --oneline -10'
```

The default task is **FetchReach** (10-second time budget, ~30 experiments/hour). For harder tasks or different simulators, see the next section.

## Different tasks

Want to try a harder task? Use `setup_task.py` to create a separate experiment directory:

```bash
# List available templates
python setup_task.py --list

# Set up an experiment directory
python setup_task.py mujoco/fetchpush my_experiment
# or: python setup_task.py mujoco/fetchpickplace my_experiment
# or: python setup_task.py isaac/fetchreach my_experiment

cd my_experiment
git init && git add -A && git commit -m "init"
uv sync
uv run prepare.py
claude --dangerously-skip-permissions
```

### Available tasks

| Template | Task | Family | Success metric | Time budget |
|----------|------|--------|----------------|-------------|
| `mujoco/fetchreach` | Reach a target position | goal-conditioned | distance < 0.05 | 10 seconds |
| `mujoco/fetchpush` | Push a cube to a goal | goal-conditioned | distance < 0.05 | 10 minutes |
| `mujoco/fetchpickplace` | Pick and place an object | goal-conditioned | distance < 0.05 | 30 minutes |
| `isaac/fetchreach` | Reach (Isaac Sim) | goal-conditioned (prototype) | distance < 0.05 | 60 seconds |
| `gym/pendulum` | Swing-up & balance | classic-control | episode return ≥ -200 | 60 seconds |
| `libero/lift` | Pick & lift (LIBERO) | image+proprio (prototype) | terminal reward = 1 | 10 minutes |

The `Family` column matters because the **eval contract differs per family**. Goal-conditioned templates use `core/evaluate.py` (distance-to-goal). Other families ship their own `evaluate.py` override.

## Adding a new template (the harness contract)

A template lives at `templates/<sim>/<task>/` and provides any subset of these files. Files **not** provided fall back to `core/`.

| File | Required? | When to override |
|------|-----------|------------------|
| `prepare.py` | yes | always — defines `ENV_ID`, `TIME_BUDGET`, `make_env`, `flatten_obs`, `get_obs_dim`, `get_action_dim`, `get_action_bounds` |
| `evaluate.py` | only if your env doesn't fit `obs["achieved_goal"]/obs["desired_goal"]` | classic-control, image-based, or any non-goal-conditioned env |
| `program.md.template` | only if the agent needs different instructions (e.g. "rewrite train.py because HER doesn't apply") | non-default algorithm fit |
| `train.py` | only if the SAC+HER baseline is wildly wrong for your env | rare — usually let the agent rewrite it as exp 1 |
| `pyproject.toml` | only if you need extra deps (e.g. robosuite, isaac-lab) | for non-Gymnasium envs |

`prepare.py` must export this exact surface (used by `core/train.py`):

```python
TIME_BUDGET: int                    # seconds
ENV_ID: str                         # informative; the harness can ignore it
def make_env(env_id=ENV_ID, render_mode=None) -> gym.Env
def flatten_obs(obs_dict) -> np.ndarray              # vector input to the policy
def get_obs_dim(env_id=ENV_ID) -> int
def get_action_dim(env_id=ENV_ID) -> int
def get_action_bounds(env_id=ENV_ID) -> (np.ndarray, np.ndarray)
```

If your env is not goal-conditioned, wrap it so `obs` is always a dict with at least `{"observation": np.ndarray}`. See `templates/gym/pendulum/prepare.py` for a working shim.

`evaluate.py` (when overridden) must export:

```python
EVAL_EPISODES: int
RENDER_EPISODES: int
def evaluate(policy_fn, env_id=ENV_ID, n_episodes=EVAL_EPISODES) -> dict   # keys: success_rate, mean_reward, mean_distance, per_episode
def render_episodes(policy_fn, env_id=ENV_ID, n_episodes=RENDER_EPISODES, output_dir="./renders", show_window=False) -> dict
```

Keep the **return-dict shape identical** to `core/evaluate.py` so `train.py` works unchanged. The third metric (`mean_distance`) is repurposed per family — for Pendulum it's mean `|theta_dot|` at episode end; for image-based tasks it could be mean Cartesian error.

A template is automatically marked `(prototype)` in `setup_task.py --list` if its `prepare.py` contains `NotImplementedError` (see `templates/libero/lift/`).

## Custom program.md

The `program.md` file is essentially a super lightweight "skill" -- it tells the agent how to run experiments. You can write your own:

```bash
# Option A: replace program.md in the repo root
cp my_custom_program.md program.md

# Option B: use --program with setup_task.py
python setup_task.py mujoco/fetchpush my_experiment --program my_custom_program.md
```

## Results

**FetchReach:**

![FetchReach Results](assets/results_plot_fetchreach.png)

**FetchPush, FetchPickPlace**: TBD Soon.

**VLA Experiments**: TBD after getting compute credits. (Support would be appreciated! [buymeacoffee.com/jellyheadandrew](https://buymeacoffee.com/jellyheadandrew))

## What changed from the original

[autoresearch](https://github.com/karpathy/autoresearch) targets LLM training, where the only feedback is loss curves. Robotics has a visual component: you can *see* what the robot is doing wrong.

Key adaptations:

- **Visual feedback loop.** After each experiment, MuJoCo renders the robot's behavior. The coding agent analyzes the rendered frames, getting qualitative feedback ("the arm overshoots and oscillates") alongside quantitative metrics -- not just numbers.
- **MuJoCo + Gymnasium Robotics** instead of nanoGPT. SAC + HER as the baseline RL algorithm.
- **Template system.** Multiple tasks and simulators in one repo. `setup_task.py` assembles flat, self-contained experiment directories.
- **Simulator modularity.** MuJoCo is fully supported; Isaac Sim support is prototyped for future use.

## Project structure

```
prepare.py                   -- env factory, obs utilities (default: FetchReach)
evaluate.py                  -- evaluation harness, rendering pipeline
train.py                     -- SAC+HER policy, training loop (agent modifies this)
program.md                   -- agent instructions
pyproject.toml               -- dependencies

core/                        -- shared source files for the template system
                                (default goal-conditioned harness)
templates/
  mujoco/                    -- MuJoCo Fetch templates (goal-conditioned)
    fetchreach/prepare.py    -- FetchReach-v4, 10s budget
    fetchpush/prepare.py     -- FetchPush-v4, 10min budget
    fetchpickplace/prepare.py -- FetchPickAndPlace-v4, 30min budget
  gym/                       -- Gymnasium classic-control templates
    pendulum/                -- prepare.py + evaluate.py + program.md.template
                                override (non-goal-conditioned harness)
  isaac/                     -- Isaac Sim templates (prototype)
    fetchreach/              -- prepare.py, evaluate.py, pyproject.toml overrides
  libero/                    -- LIBERO / robosuite templates (prototype)
    lift/                    -- prepare.py stub (NotImplementedError)
setup_task.py                -- assembles template -> experiment directory
```

## Credits

Built on [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy. Uses [Gymnasium Robotics](https://robotics.farama.org/) with [MuJoCo](https://mujoco.org/).

## License

MIT
