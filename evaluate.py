import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio
import env  # registers FishingRL-v0
from env import BAR_HALF_HEIGHT

BASE_DIR   = os.path.dirname(__file__)
FPS        = 20
TRACK_W    = 0.18
TRACK_X0   = 0.41


def _render_frame(obs, step: int, step_reward: float, total_reward: float,
                  agent_label: str = "Random", episode: int = 1,
                  n_episodes: int = 1) -> np.ndarray:
    fish_y, bar_y, bar_vel, progress, weather, fish_type, season_idx = obs.tolist()

    fig = plt.figure(figsize=(9, 5), facecolor="#0d1b2a")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.8], wspace=0.35,
                          left=0.06, right=0.97, top=0.92, bottom=0.08)

    # ── Left: vertical fishing track ──────────────────────────────────────────
    ax_track = fig.add_subplot(gs[0])
    ax_track.set_facecolor("#1b2838")
    ax_track.set_xlim(0, 1)
    ax_track.set_ylim(0, 1)
    ax_track.set_xticks([])
    ax_track.set_yticks(np.linspace(0, 1, 6))
    ax_track.tick_params(colors="white", labelsize=8)
    ax_track.set_title("Fishing Track", color="white", fontsize=11, pad=6)
    for sp in ax_track.spines.values():
        sp.set_color("#4a6fa5")

    # guide line (center of track)
    ax_track.axvline(0.5, color="#4a6fa5", lw=0.8, ls="--", alpha=0.5)

    # bar (green rectangle)
    bar_lo = bar_y - BAR_HALF_HEIGHT
    bar_hi = bar_y + BAR_HALF_HEIGHT
    bar_rect = patches.FancyBboxPatch(
        (TRACK_X0, bar_lo), TRACK_W, bar_hi - bar_lo,
        boxstyle="round,pad=0.01",
        linewidth=1.2, edgecolor="#76c442", facecolor="#4CAF50", alpha=0.85,
    )
    ax_track.add_patch(bar_rect)
    ax_track.text(0.5, bar_y, "BAR", ha="center", va="center",
                  color="white", fontsize=7, fontweight="bold")

    # fish (circle)
    fish_in_bar = (bar_lo <= fish_y <= bar_hi)
    fish_color = "#FFD700" if fish_in_bar else "#FF6B6B"
    ax_track.plot(0.5, fish_y, "o", color=fish_color, markersize=14,
                  markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    ax_track.text(0.5, fish_y, "F", ha="center", va="center",
                  color="white", fontsize=7, fontweight="bold", zorder=6)

    # catch / miss label
    status_txt = "CATCH" if fish_in_bar else "MISS"
    status_col = "#4CAF50" if fish_in_bar else "#F44336"
    ax_track.text(0.5, 0.03, status_txt, ha="center", va="bottom",
                  color=status_col, fontsize=9, fontweight="bold")

    # ── Right: info panel ─────────────────────────────────────────────────────
    ax_info = fig.add_subplot(gs[1])
    ax_info.set_facecolor("#1b2838")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")
    ep_tag = f"Ep {episode}/{n_episodes}" if n_episodes > 1 else ""
    title = f"Episode Info  [{agent_label}]  {ep_tag}".strip()
    ax_info.set_title(title, color="white", fontsize=11, pad=6)

    # Progress bar background
    ax_info.add_patch(patches.Rectangle(
        (0.05, 0.84), 0.9, 0.07, facecolor="#2e3f50", edgecolor="#4a6fa5", linewidth=1.2))
    prog_color = "#4CAF50" if progress > 0.6 else "#FF9800" if progress > 0.3 else "#F44336"
    ax_info.add_patch(patches.Rectangle(
        (0.05, 0.84), 0.9 * progress, 0.07, facecolor=prog_color))
    ax_info.text(0.5, 0.875, f"Catch Progress  {progress:.2f}",
                 ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    # Info rows
    weather_str = "Rainy  (1.3x speed)" if int(weather) == 1 else "Sunny"
    fish_str    = "Rare  (+20 on catch)" if int(fish_type) == 1 else "Normal  (+10 on catch)"
    rows = [
        ("Weather",      weather_str,          "#64b5f6"),
        ("Fish Type",    fish_str,             "#ce93d8"),
        ("Bar Velocity", f"{bar_vel:+.3f}",   "#ffffff"),
        ("Step",         f"{step}",            "#ffffff"),
        ("Step Reward",  f"{step_reward:+.2f}", "#4CAF50" if step_reward > 0 else "#F44336"),
        ("Total Reward", f"{total_reward:+.2f}", "#FFD700"),
    ]
    y0 = 0.72
    for label, value, vcol in rows:
        ax_info.text(0.05, y0, label, color="#aab8c2", fontsize=9, va="top")
        ax_info.text(0.95, y0, value, color=vcol, fontsize=9, va="top", ha="right",
                     fontweight="bold")
        ax_info.axhline(y0 - 0.005, xmin=0.05, xmax=0.95, color="#2e3f50", lw=0.6)
        y0 -= 0.095

    # Convert to RGB numpy array
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]  # RGBA → RGB
    plt.close(fig)
    return frame


def _run_episode(env_: gym.Env, policy=None, agent_label: str = "Random",
                 seed: int = 42, episode: int = 1,
                 n_episodes: int = 1) -> tuple[list[np.ndarray], int, float]:
    """Run one episode, return (frames, steps, total_reward).

    policy: callable(obs) -> action, or None for random.
    """
    obs, _ = env_.reset(seed=seed)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    step = 0
    last_reward = 0.0

    frames.append(_render_frame(obs, step, last_reward, total_reward,
                                agent_label, episode, n_episodes))

    while True:
        action = policy(obs) if policy is not None else env_.action_space.sample()
        obs, reward, terminated, truncated, _ = env_.step(action)
        step += 1
        total_reward += reward
        last_reward = float(reward)
        frames.append(_render_frame(obs, step, last_reward, total_reward,
                                    agent_label, episode, n_episodes))
        if terminated or truncated:
            break

    return frames, step, total_reward


def _write_video(frames: list[np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def record_random_agent(seed: int = 42) -> None:
    output = os.path.join(BASE_DIR, "random_agent_video.mp4")
    e = gym.make("FishingRL-v0")
    frames, step, total_reward = _run_episode(e, policy=None,
                                              agent_label="Random", seed=seed)
    e.close()
    _write_video(frames, output)
    print(f"[random]  steps={step}  total_reward={total_reward:.2f}")
    print(f"video saved → {output}")


def record_trained_agent(model_path: str | None = None, seed: int = 42,
                         n_episodes: int = 10) -> None:
    from stable_baselines3 import PPO

    if model_path is None:
        model_path = os.path.join(BASE_DIR, "models", "ppo_fishing_seed_42.zip")

    model = PPO.load(model_path)

    def policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    output = os.path.join(BASE_DIR, "trained_agent_video.mp4")
    e = gym.make("FishingRL-v0")
    all_frames: list[np.ndarray] = []
    rewards = []

    for ep in range(1, n_episodes + 1):
        frames, step, total_reward = _run_episode(
            e, policy=policy, agent_label="PPO (trained)",
            seed=seed + ep - 1, episode=ep, n_episodes=n_episodes,
        )
        all_frames.extend(frames)
        rewards.append(total_reward)
        print(f"  ep {ep:2d}/{n_episodes}  steps={step:3d}  reward={total_reward:+.2f}")

    e.close()
    _write_video(all_frames, output)
    print(f"\n[trained] {n_episodes} episodes  "
          f"mean_reward={np.mean(rewards):.2f}  "
          f"total_frames={len(all_frames)}")
    print(f"video saved → {output}")


if __name__ == "__main__":
    record_random_agent()
    record_trained_agent()
