"""Gymnasium task and editable linear torque policy for the UR5 golf scene."""

from __future__ import annotations

import math
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from golf_env import GolfEnvironment


TORQUE_LIMIT_NM = 28.0  # joint5 effort limit in the supplied UR5 URDF
FOLLOW_THROUGH_RAD = 0.15
REWARD_OFFSET_M = 0.02

# The first four features are one-hot time windows before club contact.
# Remaining features turn on after contact (or at the 1 s swing timeout).
FEATURE_NAMES = (
    "swing_0.05_to_0.20_s",
    "swing_0.20_to_0.35_s",
    "swing_0.35_to_0.60_s",
    "swing_0.60_to_1.00_s",
    "hold_gravity_nm",
    "hold_angle_error_rad",
    "hold_angular_speed_rad_s",
)

# Matches the existing demo: +1 N m pulse, then gravity-compensated PD hold.
DEFAULT_MATRIX = np.array([[1.0], [0.0], [0.0], [0.0],
                           [1.0], [40.0], [-8.0]], dtype=np.float64)


def distance_reward(distance_m: float) -> float:
    """Terminal score; maximum is 50 at zero distance."""
    distance = float(distance_m)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("distance_m must be finite and nonnegative")
    return 1.0 / (distance + REWARD_OFFSET_M)


class MatrixGolfPolicy:
    """A ProPS-style state-feature-by-action matrix, W (7 x 1)."""

    def __init__(self, matrix: Sequence[Sequence[float]] | np.ndarray = DEFAULT_MATRIX):
        self.update(matrix)

    def update(self, matrix: Sequence[Sequence[float]] | np.ndarray) -> None:
        candidate = np.asarray(matrix, dtype=np.float64)
        if candidate.shape != (len(FEATURE_NAMES), 1):
            raise ValueError(f"matrix must have shape ({len(FEATURE_NAMES)}, 1)")
        if not np.all(np.isfinite(candidate)):
            raise ValueError("matrix entries must be finite")
        self.matrix = candidate.copy()

    def action(self, observation: np.ndarray) -> np.ndarray:
        features = np.asarray(observation, dtype=np.float64)
        if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
            raise ValueError("observation must be a finite feature vector")
        torque = float(features @ self.matrix[:, 0])
        return np.array([np.clip(torque, -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)],
                        dtype=np.float32)


class GolfRLEnv(gym.Env):
    """One golf shot; action is wrist torque held for ``frame_skip`` ms.

    The five other UR5 joints remain locked by the scene's equality
    constraints. The policy controls joint5 only. Reward is terminal so
    repeated slow/no-hit states cannot accumulate a large score.
    """

    metadata = {"render_modes": []}

    def __init__(self, duration_s: float = 5.0, frame_skip: int = 20,
                 render_mode: None = None):
        super().__init__()
        if render_mode is not None:
            raise ValueError("GolfRLEnv is headless; use run_golf.py for replay")
        self.render_mode = render_mode
        if not math.isfinite(duration_s) or duration_s <= 1.0:
            raise ValueError("duration_s must be finite and greater than 1")
        if not isinstance(frame_skip, int) or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        self.sim = GolfEnvironment()
        self.duration_s = float(duration_s)
        self.frame_skip = frame_skip
        self.action_space = spaces.Box(-TORQUE_LIMIT_NM, TORQUE_LIMIT_NM,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(len(FEATURE_NAMES),),
                                            dtype=np.float32)
        self.hold_angle_rad: float | None = None
        self._done = False

    def _features(self) -> np.ndarray:
        observation = self.sim.observe()
        t = observation["time_s"]
        features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        if self.hold_angle_rad is None:
            for i, (start, stop) in enumerate(((0.05, 0.20), (0.20, 0.35),
                                               (0.35, 0.60), (0.60, 1.00))):
                features[i] = float(start <= t < stop)
        else:
            dof = int(self.sim.model.joint("joint5").dofadr[0])
            features[4] = float(self.sim.data.qfrc_bias[dof])
            features[5] = self.hold_angle_rad - observation["swing_angle_rad"]
            features[6] = observation["swing_speed_rad_s"]
        return features

    def _info(self) -> dict:
        observation = self.sim.observe()
        return {
            "distance_to_hole_m": observation["distance_to_hole_m"],
            "closest_distance_m": observation["closest_distance_m"],
            "ball_hit": observation["ball_hit"],
            "ball_sunk": observation["ball_sunk"],
            "ball_position_m": observation["ball_position_m"],
            "wrist_angle_deg": observation["wrist_angle_deg"],
            "wrist_speed_rad_s": observation["swing_speed_rad_s"],
            "time_s": observation["time_s"],
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.sim.reset()
        self.hold_angle_rad = None
        self._done = False
        return self._features(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("episode ended; call reset() before step()")
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (1,) or not np.all(np.isfinite(values)):
            raise ValueError("action must contain one finite torque")
        torque = float(np.clip(values[0], -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM))
        for _ in range(self.frame_skip):
            observation = self.sim.step(torque)
            if self.hold_angle_rad is None and (observation["ball_hit"] or
                                                observation["time_s"] >= 1.0):
                self.hold_angle_rad = observation["swing_angle_rad"] + (
                    FOLLOW_THROUGH_RAD if observation["ball_hit"] else 0.0)
            if observation["time_s"] >= self.duration_s:
                break
        info = self._info()
        terminated = bool(info["ball_sunk"])
        truncated = not terminated and info["time_s"] >= self.duration_s
        self._done = terminated or truncated
        reward = distance_reward(info["distance_to_hole_m"]) if self._done else 0.0
        return self._features(), reward, terminated, truncated, info

    def close(self):
        self.sim = None


# Allows gym.make("golf_rl_env:GolfSwing-v0") when simulationsetup is on PYTHONPATH.
gym.register(id="GolfSwing-v0", entry_point="golf_rl_env:GolfRLEnv")
