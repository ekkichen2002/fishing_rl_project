"""Read Monitor logs for 3 seeds and plot raw + smoothed episode reward curves."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
LOG_DIR  = os.path.join(BASE_DIR, "logs")
OUT_PATH = os.path.join(BASE_DIR, "training_curve.png")

SEEDS   = [42, 123, 456]
COLORS  = ["#4FC3F7", "#81C784", "#FFB74D"]   # blue, green, orange
SMOOTH  = 50   # rolling window for smoothed curve


def load_monitor(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (cumulative_timesteps, episode_rewards) from a Monitor CSV."""
    path = os.path.join(LOG_DIR, f"seed_{seed}.monitor.csv")
    # Monitor CSV: line 1 = #comment, line 2 = header (r,l,t), rest = data
    rewards, lengths = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 2:          # skip comment + header
                continue
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            rewards.append(float(parts[0]))
            lengths.append(int(parts[1]))

    rewards = np.array(rewards)
    lengths = np.array(lengths)
    cumsteps = np.cumsum(lengths)
    return cumsteps, rewards


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = arr[lo : i + 1].mean()
    return out


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#1b2838")
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_color("#4a6fa5")
    ax.grid(color="#2e3f50", linestyle="--", linewidth=0.6, alpha=0.7)

    for seed, color in zip(SEEDS, COLORS):
        cumsteps, rewards = load_monitor(seed)
        smoothed = rolling_mean(rewards, SMOOTH)

        # raw rewards (faint scatter)
        ax.scatter(cumsteps, rewards, s=3, color=color, alpha=0.18, linewidths=0)
        # smoothed curve
        ax.plot(cumsteps, smoothed, color=color, linewidth=1.8,
                label=f"seed {seed}  (final ≈ {smoothed[~np.isnan(smoothed)][-1]:.1f})")

    ax.set_xlabel("Total Timesteps", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title(f"PPO Training Curves — FishingRL-v0  (smooth window={SMOOTH})",
                 fontsize=12, pad=8)
    ax.axhline(0, color="#4a6fa5", linewidth=0.8, linestyle=":")

    legend = ax.legend(fontsize=9, framealpha=0.3, labelcolor="white",
                       facecolor="#1b2838", edgecolor="#4a6fa5")
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor())
    print(f"training curve saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
