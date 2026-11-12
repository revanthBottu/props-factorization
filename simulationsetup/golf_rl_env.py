"""Gymnasium two-joint golf task and regular (unfactorized) ProPS policy."""

from __future__ import annotations

import math
from typing import Sequence

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from golf_env import CLUB_ADDRESS_ANGLE_RAD, GolfEnvironment


ACTION_NAMES = ("position_joint4_torque_nm", "swing_joint5_torque_nm")
TORQUE_LIMITS_NM = np.array([28.0, 28.0], dtype=np.float64)
FOLLOW_THROUGH_RAD = 0.15

HIT_REWARD = 20.0
CENTRAL_HIT_REWARD = 30.0
DISTANCE_REWARD = 30.0
SUNK_REWARD = 20.0
SMOOTHNESS_PENALTY_WEIGHT = 0.20
COLLISION_STEP_PENALTY = 0.02

# The rows are ordinary ProPS state features. The first two let the LLM add
# open-loop positioning pulses; the next three schedule the swing after the
# positioning interval; and the remaining rows support smooth feedback holds.
FEATURE_NAMES = (
    "position_0.00_to_0.25_s",
    "position_0.25_to_0.55_s",
    "swing_0.55_to_0.70_s",
    "swing_0.70_to_0.85_s",
    "swing_0.85_to_1.10_s",
    "position_gravity_nm",
    "position_target_error_rad",
    "position_speed_rad_s",
    "swing_gravity_nm_after_shot",
    "swing_hold_error_rad_after_shot",
    "swing_speed_rad_s_after_shot",
)

# W has shape (state features, actions). Column 0 positions joint4 at -pi/2;
# column 1 applies the demonstrated joint5 pulse and follow-through hold.
DEFAULT_MATRIX = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [20.0, 0.0],
    [-6.0, 0.0],
    [0.0, 1.0],
    [0.0, 40.0],
    [0.0, -8.0],
], dtype=np.float64)


def positive_shot_reward(distance_m: float, ball_hit: bool,
                         impact_centrality: float, ball_sunk: bool) -> dict:
    """Return the terminal positive components of the task reward."""
    distance = float(distance_m)
    centrality = float(impact_centrality)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("distance_m must be finite and nonnegative")
    if not math.isfinite(centrality) or not 0.0 <= centrality <= 1.0:
        raise ValueError("impact_centrality must be between zero and one")
    hit = bool(ball_hit)
    return {
        "hit": HIT_REWARD if hit else 0.0,
        "central_hit": CENTRAL_HIT_REWARD * centrality if hit else 0.0,
        "distance": DISTANCE_REWARD / (1.0 + 4.0 * distance) if hit else 0.0,
        "sunk": SUNK_REWARD if hit and ball_sunk else 0.0,
    }


def distance_reward(distance_m: float) -> float:
    """Backward-compatible access to the distance component of a hit."""
    return positive_shot_reward(distance_m, True, 0.0, False)["distance"]


class MatrixGolfPolicy:
    """A regular ProPS linear policy, ``action = state @ W`` (no factors)."""

    def __init__(self, matrix: Sequence[Sequence[float]] | np.ndarray = DEFAULT_MATRIX):
        self.update(matrix)

    def update(self, matrix: Sequence[Sequence[float]] | np.ndarray) -> None:
        candidate = np.asarray(matrix, dtype=np.float64)
        expected = (len(FEATURE_NAMES), len(ACTION_NAMES))
        if candidate.shape != expected:
            raise ValueError(f"matrix must have shape {expected}")
        if not np.all(np.isfinite(candidate)):
            raise ValueError("matrix entries must be finite")
        self.matrix = candidate.copy()

    def action(self, observation: np.ndarray) -> np.ndarray:
        features = np.asarray(observation, dtype=np.float64)
        if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
            raise ValueError("observation must be a finite feature vector")
        torques = features @ self.matrix
        return np.clip(torques, -TORQUE_LIMITS_NM, TORQUE_LIMITS_NM).astype(
            np.float32
        )


class GolfRLEnv(gym.Env):
    """Deterministic two-joint golf shot with Gymnasium's current API.

    Joint4 must first position the club from the universal reset pose and
    joint5 must swing. Positive shot quality is terminal; smoothness and
    collision penalties are returned incrementally. ``info['reward_components']``
    exposes the complete episode breakdown at every step.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, duration_s: float = 5.0, frame_skip: int = 20,
                 render_mode: str | None = None, camera: str = "swing"):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if not math.isfinite(duration_s) or duration_s <= 1.1:
            raise ValueError("duration_s must be finite and greater than 1.1")
        if not isinstance(frame_skip, int) or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        self.render_mode = render_mode
        self.camera = camera
        self.sim = GolfEnvironment()
        self.duration_s = float(duration_s)
        self.frame_skip = frame_skip
        self.action_space = spaces.Box(
            low=-TORQUE_LIMITS_NM.astype(np.float32),
            high=TORQUE_LIMITS_NM.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(FEATURE_NAMES),), dtype=np.float32
        )
        self._renderer = None
        self._done = False

    def _features(self) -> np.ndarray:
        observation = self.sim.observe()
        t = observation["time_s"]
        features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        features[0] = float(0.00 <= t < 0.25)
        features[1] = float(0.25 <= t < 0.55)
        if self.hold_angle_rad is None:
            for index, (start, stop) in enumerate(
                ((0.55, 0.70), (0.70, 0.85), (0.85, 1.10)), start=2
            ):
                features[index] = float(start <= t < stop)

        position_dof = int(self.sim.model.joint("joint4").dofadr[0])
        features[5] = float(self.sim.data.qfrc_bias[position_dof])
        features[6] = float(observation["position_error_rad"])
        features[7] = float(observation["position_speed_rad_s"])
        if self.hold_angle_rad is not None:
            swing_dof = int(self.sim.model.joint("joint5").dofadr[0])
            features[8] = float(self.sim.data.qfrc_bias[swing_dof])
            features[9] = self.hold_angle_rad - observation["swing_angle_rad"]
            features[10] = float(observation["swing_speed_rad_s"])
        return features

    def _reward_components(self) -> dict:
        observation = self.sim.observe()
        positive = self._terminal_components.copy()
        return {
            **positive,
            "smoothness_penalty": -self._smoothness_penalty,
            "collision_penalty": -self._collision_penalty,
            "total": (
                sum(positive.values())
                - self._smoothness_penalty
                - self._collision_penalty
            ),
        }

    def _info(self) -> dict:
        observation = self.sim.observe()
        return {
            **observation,
            "club_address_target_rad": CLUB_ADDRESS_ANGLE_RAD,
            "reward_components": self._reward_components(),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.sim.reset()
        self.hold_angle_rad: float | None = None
        self._previous_action = np.zeros(2, dtype=np.float64)
        self._smoothness_penalty = 0.0
        self._collision_penalty = 0.0
        self._last_collision_steps = 0
        self._terminal_components = {
            "hit": 0.0, "central_hit": 0.0, "distance": 0.0, "sunk": 0.0
        }
        self._done = False
        return self._features(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("episode ended; call reset() before step()")
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("action must contain two finite torques")
        values = np.clip(values, -TORQUE_LIMITS_NM, TORQUE_LIMITS_NM)

        normalized_delta = (values - self._previous_action) / (2 * TORQUE_LIMITS_NM)
        smoothness_delta = SMOOTHNESS_PENALTY_WEIGHT * float(
            normalized_delta @ normalized_delta
        )
        self._smoothness_penalty += smoothness_delta
        self._previous_action = values.copy()

        for _ in range(self.frame_skip):
            observation = self.sim.step(values)
            if self.hold_angle_rad is None and (
                observation["ball_hit"] or observation["time_s"] >= 1.10
            ):
                self.hold_angle_rad = observation["swing_angle_rad"] + (
                    FOLLOW_THROUGH_RAD if observation["ball_hit"] else 0.0
                )
            if observation["time_s"] >= self.duration_s:
                break

        new_collision_steps = self.sim.collision_steps - self._last_collision_steps
        collision_delta = COLLISION_STEP_PENALTY * new_collision_steps
        self._collision_penalty += collision_delta
        self._last_collision_steps = self.sim.collision_steps

        info = self._info()
        terminated = bool(info["ball_sunk"])
        truncated = not terminated and info["time_s"] >= self.duration_s
        self._done = terminated or truncated
        reward = -smoothness_delta - collision_delta
        if self._done:
            self._terminal_components = positive_shot_reward(
                info["distance_to_hole_m"], info["ball_hit"],
                info["impact_centrality"], info["ball_sunk"]
            )
            reward += sum(self._terminal_components.values())
            info = self._info()
        return self._features(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.sim.model, height=480, width=640
            )
        self._renderer.update_scene(self.sim.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.sim = None


# This module path works with gym.make("golf_rl_env:GolfSwing-v0") and with
# the ContinualSpaceGeneralWorld wrapper used by props-factorization.
gym.register(id="GolfSwing-v0", entry_point="golf_rl_env:GolfRLEnv")
