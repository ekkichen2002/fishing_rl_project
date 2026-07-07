import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

GRAVITY = 0.03
THRUST = 0.07
V_MAX = 0.1
BAR_HALF_HEIGHT = 0.1
DELTA_POS = 0.02
DELTA_NEG = 0.03
MAX_STEPS = 500
BASE_FISH_SPEED = 0.02
RAINY_SPEED_MULT = 1.3

# Season definitions
SEASONS = ["spring", "summer", "autumn", "winter"]

# Per-season rare fish probability: (sunny_prob, rainy_prob)
SEASON_RARE_PROB = {
    "spring": (0.20, 0.40),
    "summer": (0.30, 0.50),
    "autumn": (0.25, 0.45),
    "winter": (0.10, 0.30),
}

# Per-season base fish speed multiplier
SEASON_SPEED_MULT = {
    "spring": 1.0,
    "summer": 1.1,
    "autumn": 0.9,
    "winter": 0.8,
}


class FishingRLEnv(gym.Env):
    """Stardew Valley-inspired fishing minigame RL environment.

    Observation (7-dim float32):
        [fish_y, bar_y, bar_velocity, catch_progress, weather, fish_type, season]

    Actions:
        0 = release (gravity pulls bar down)
        1 = hold   (thrust pushes bar up)

    Args:
        season (str | None): fix season to one of 'spring','summer','autumn','winter'.
                             If None, season is randomised each episode.
        render_mode (str | None): 'rgb_array' or None.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, season: str | None = None, render_mode: str | None = None):
        super().__init__()

        if season is not None and season not in SEASONS:
            raise ValueError(f"season must be one of {SEASONS}, got {season!r}")
        self._fixed_season = season
        self.render_mode = render_mode

        # obs: [fish_y, bar_y, bar_velocity, catch_progress, weather, fish_type, season]
        self.observation_space = spaces.Box(
            low=np.array( [0.0, 0.0, -V_MAX, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0,  V_MAX, 1.0, 1.0, 1.0, 3.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        # Internal state
        self.fish_y = 0.5
        self.bar_y = 0.5
        self.bar_velocity = 0.0
        self.catch_progress = 0.5
        self.weather = 0       # 0=sunny, 1=rainy
        self.fish_type = 0     # 0=normal, 1=rare
        self.season_idx = 0    # index into SEASONS
        self._fish_target = 0.5
        self._fish_vel = 0.0
        self._step = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0

        # Season: fixed or random each episode
        if self._fixed_season is not None:
            self.season_idx = SEASONS.index(self._fixed_season)
        else:
            self.season_idx = int(self.np_random.integers(0, 4))
        season_name = SEASONS[self.season_idx]

        self.weather = int(self.np_random.integers(0, 2))
        rare_prob = SEASON_RARE_PROB[season_name][self.weather]
        self.fish_type = int(self.np_random.random() < rare_prob)

        self.fish_y = float(self.np_random.uniform(0.1, 0.9))
        self.bar_y = float(self.np_random.uniform(BAR_HALF_HEIGHT, 1.0 - BAR_HALF_HEIGHT))
        self.bar_velocity = 0.0
        self.catch_progress = 0.5
        self._fish_target = float(self.np_random.uniform(0.0, 1.0))
        self._fish_vel = 0.0
        return self._obs(), {}

    def step(self, action):
        self._step += 1

        # Bar physics
        if action == 1:
            self.bar_velocity += THRUST
        else:
            self.bar_velocity -= GRAVITY
        self.bar_velocity = float(np.clip(self.bar_velocity, -V_MAX, V_MAX))
        self.bar_y = float(
            np.clip(self.bar_y + self.bar_velocity, BAR_HALF_HEIGHT, 1.0 - BAR_HALF_HEIGHT)
        )

        # Fish movement
        self._step_fish()

        # Determine overlap
        bar_lo = self.bar_y - BAR_HALF_HEIGHT
        bar_hi = self.bar_y + BAR_HALF_HEIGHT
        fish_in_bar = bar_lo <= self.fish_y <= bar_hi

        # Progress and step reward
        if fish_in_bar:
            self.catch_progress = min(1.0, self.catch_progress + DELTA_POS)
            reward = 0.1
        else:
            self.catch_progress = max(0.0, self.catch_progress - DELTA_NEG)
            reward = -0.2

        # Terminal bonus / penalty
        terminated = False
        if self.catch_progress >= 1.0:
            terminated = True
            reward += 20.0 if self.fish_type == 1 else 10.0
        elif self.catch_progress <= 0.0:
            terminated = True
            reward += -5.0

        truncated = self._step >= MAX_STEPS

        return self._obs(), float(reward), terminated, truncated, {}

    def render(self):
        """Return an RGB frame (H x W x 3 uint8) when render_mode='rgb_array'."""
        if self.render_mode != "rgb_array":
            return None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(3, 6))
        fig.patch.set_facecolor("#0d1b2a")
        ax.set_facecolor("#1b2838")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_title(
            f"Season:{SEASONS[self.season_idx]}  {'Rain' if self.weather else 'Sun'}  "
            f"{'Rare' if self.fish_type else 'Normal'}",
            color="white", fontsize=8,
        )

        # Bar
        bar_lo = self.bar_y - BAR_HALF_HEIGHT
        ax.add_patch(patches.Rectangle(
            (0.3, bar_lo), 0.4, BAR_HALF_HEIGHT * 2,
            facecolor="#4CAF50", edgecolor="#76c442", linewidth=1.2,
        ))

        # Fish
        fish_in = (bar_lo <= self.fish_y <= self.bar_y + BAR_HALF_HEIGHT)
        ax.plot(0.5, self.fish_y, "o",
                color="#FFD700" if fish_in else "#FF6B6B",
                markersize=12, markeredgecolor="white", zorder=5)

        # Progress bar
        prog_color = "#4CAF50" if self.catch_progress > 0.6 else \
                     "#FF9800" if self.catch_progress > 0.3 else "#F44336"
        ax.add_patch(patches.Rectangle((0.05, 0.01), 0.9 * self.catch_progress, 0.025,
                                       facecolor=prog_color))
        ax.add_patch(patches.Rectangle((0.05, 0.01), 0.9, 0.025,
                                       facecolor="none", edgecolor="#4a6fa5", linewidth=0.8))

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        return frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self):
        return np.array(
            [self.fish_y, self.bar_y, self.bar_velocity, self.catch_progress,
             float(self.weather), float(self.fish_type), float(self.season_idx)],
            dtype=np.float32,
        )

    def _step_fish(self):
        season_name = SEASONS[self.season_idx]
        speed = (BASE_FISH_SPEED
                 * SEASON_SPEED_MULT[season_name]
                 * (RAINY_SPEED_MULT if self.weather == 1 else 1.0))
        # Rare fish is faster and jitterier
        if self.fish_type == 1:
            speed *= 1.25

        diff = self._fish_target - self.fish_y
        self._fish_vel += np.sign(diff) * speed * 0.4
        self._fish_vel = float(np.clip(self._fish_vel, -speed, speed))
        self.fish_y = float(np.clip(self.fish_y + self._fish_vel, 0.0, 1.0))

        # Pick new target when close or randomly (rare fish switches more often)
        switch_prob = 0.08 if self.fish_type == 1 else 0.05
        if abs(self.fish_y - self._fish_target) < 0.04 or self.np_random.random() < switch_prob:
            self._fish_target = float(self.np_random.uniform(0.0, 1.0))


register(id="FishingRL-v0", entry_point="env:FishingRLEnv")
