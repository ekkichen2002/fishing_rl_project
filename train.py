"""PPO training for FishingRL-v0.

Usage:
    python train.py                  # full run: 500k steps × 3 seeds
    python train.py --steps 50000    # quick validation run (1 seed)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import env  # registers FishingRL-v0

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
LOG_DIR   = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEEDS       = [42, 123, 456]
TOTAL_STEPS = 500_000
GAMMA       = 0.99

# avg episode ≈ 24 steps; n_steps=256 ≈ 10 episodes per rollout
N_STEPS     = 256
BATCH_SIZE  = 64   # must divide N_STEPS
N_EPOCHS    = 10
LR          = 3e-4
ENT_COEF    = 0.01  # light entropy bonus to encourage exploration
CKPT_FREQ   = 50_000


def _make_env(seed: int, season: str | None = None) -> gym.Env:
    # Pass season=None to randomise per episode, or fix e.g. season="summer"
    e = gym.make("FishingRL-v0", season=season)
    log_file = os.path.join(LOG_DIR, f"seed_{seed}")
    e = Monitor(e, filename=log_file)
    return e


def train_seed(seed: int, total_steps: int) -> None:
    for d in (LOG_DIR, MODEL_DIR, CKPT_DIR):
        os.makedirs(d, exist_ok=True)

    environment = _make_env(seed, season=None)  # None = random season per episode

    checkpoint_cb = CheckpointCallback(
        save_freq=CKPT_FREQ,
        save_path=os.path.join(CKPT_DIR, f"seed_{seed}"),
        name_prefix="ppo_fishing",
        verbose=0,
    )

    model = PPO(
        "MlpPolicy",
        environment,
        seed=seed,
        gamma=GAMMA,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        ent_coef=ENT_COEF,
        verbose=1,
    )

    print(f"\n{'='*55}")
    print(f"  Training  seed={seed}  total_steps={total_steps:,}")
    print(f"{'='*55}")

    model.learn(total_timesteps=total_steps, callback=checkpoint_cb,
                reset_num_timesteps=True)

    save_path = os.path.join(MODEL_DIR, f"ppo_fishing_seed_{seed}")
    model.save(save_path)
    environment.close()

    print(f"\n  Model saved → {save_path}.zip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
                        help="Timesteps per seed (default 500 000)")
    args = parser.parse_args()

    seeds = SEEDS[:1] if args.steps < TOTAL_STEPS else SEEDS

    for seed in seeds:
        train_seed(seed, args.steps)

    print("\nAll training runs complete.")


if __name__ == "__main__":
    main()
