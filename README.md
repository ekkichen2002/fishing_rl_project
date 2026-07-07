# Fishing RL — A Reinforcement Learning Formulation

A custom Gymnasium environment inspired by Stardew Valley's fishing minigame, solved with PPO (Stable-Baselines3).

Built for NTU **CA6126 — Reinforcement Learning** Final Project.

## Team

Chen Siyi · Luo Chenxi · Shi Yiheng

## Game Description

The agent controls a vertical bar trying to stay aligned with a moving fish:

- **Fishing Track**: a vertical track `[0, 1]`. The fish moves randomly up and down — faster in rainy weather, and rare fish jitter more.
- **Fishing Bar**: the agent holds to thrust the bar upward, releases to let it fall under gravity, trying to keep the bar aligned with the fish.
- **Catch Progress**: when the bar overlaps the fish, progress rises (+0.02/step); when it misses, progress drops faster (−0.03/step). Filling the bar to 1.0 wins the episode.
- **Context variables per episode**: Season (Spring/Summer/Autumn/Winter), Weather (Sunny/Rainy, +30% fish speed), Fish Type (Normal/Rare, reward ×2).

## MDP Formulation

**State space** (7-dim, float32): `fish_y`, `bar_y`, `bar_velocity`, `catch_progress`, `weather`, `fish_type`, `season`

**Action space** (Discrete-2): `0 = Release` (gravity −0.03/step), `1 = Hold` (thrust +0.07/step)

**Reward function**:
| Event | Reward |
|---|---|
| Fish inside bar (per step) | +0.1 |
| Fish outside bar (per step) | −0.2 |
| Success: normal fish caught | +10.0 |
| Success: rare fish caught | +20.0 |
| Failure: fish escapes | −5.0 |

The step penalty (−0.2) outweighing the step reward (+0.1) discourages passive behaviour; rare fish double the terminal reward.

## RL Algorithm — PPO

Chosen for its actor-critic MLP fit to the 7-dim continuous observation + 2-action discrete space, dense per-step reward signal, and stable clipped-surrogate training.

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (Stable-Baselines3) |
| Policy | MlpPolicy |
| Total timesteps | 500,000 × 3 seeds |
| n_steps | 256 |
| batch_size | 64 |
| n_epochs | 10 |
| learning_rate | 3 × 10⁻⁴ |
| gamma (γ) | 0.99 |
| ent_coef | 0.01 |
| Seeds | 42, 123, 456 |

## Results

All 3 seeds converge from a random baseline of ~−8.5 mean reward to a stable plateau of ~14–17 within 35k steps, holding through the full 500k steps:

- Seed 42 → 15.53
- Seed 123 → 14.76
- Seed 456 → 15.95

**Random agent**: ~22 steps before failure, total reward −8.5, erratic bar movement, no directed control.

**Trained agent (PPO, 500k steps)**: mean reward +15.42 over 10 episodes, consistently tracks fish position, fills progress to 1.0 in ~28–38 steps, handles both normal and rare fish.

See `training_curve.png` for the full learning curves, and `random_agent_video.mp4` / `trained_agent_video.mp4` for side-by-side behavioural comparison.

## Files

- `env.py` — custom Gymnasium environment (physics, fish movement, reward logic)
- `train.py` — PPO training script (Stable-Baselines3)
- `evaluate.py` — evaluation script comparing random vs. trained policy
- `plot_training_curve.py` — generates the training curve visualisation
- `logs/` — raw training monitor logs per seed
- `requirements.txt` — dependencies

## Setup

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```
