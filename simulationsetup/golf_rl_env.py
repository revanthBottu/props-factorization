"""Gymnasium six-joint golf task and regular (unfactorized) ProPS policy."""

from __future__ import annotations

import math
from typing import Sequence

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from golf_env import ACTION_JOINT_NAMES, GolfEnvironment


N_JOINTS = 6
ACTION_NAMES = tuple(f"joint{index}_torque_nm" for index in range(N_JOINTS))
TORQUE_LIMITS_NM = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
N_TIME_PHASES = 5

CONTACT_REWARD = 10.0
LAUNCH_SPEED_REWARD = 10.0
LAUNCH_DIRECTION_REWARD = 10.0
TARGET_LAUNCH_SPEED_M_S = 2.0
CENTRAL_HIT_REWARD = 40.0
DISTANCE_REWARD = 40.0
SUNK_REWARD = 30.0
PROXIMITY_REWARD_MAX = 8.0
SMOOTHNESS_PENALTY_WEIGHT = 0.20
EFFORT_PENALTY_WEIGHT = 0.005
# Any robot/club contact other than the intended club-head/ball contact is a
# serious sim-to-real safety failure. A brief new pair costs 20 points, while
# sustained contact costs another 50 points per second. The cap exceeds the
# complete 140-point terminal shot reward, so collision-heavy solutions cannot
# outperform physically clean ones.
COLLISION_EVENT_PENALTY = 20.0
COLLISION_TIME_PENALTY_PER_S = 50.0
COLLISION_PENALTY_CAP = 250.0
BALL_SETTLE_SPEED_M_S = 0.03
BALL_SETTLE_TIME_S = 0.50

# The controller receives generic state only: no address pose, inverse
# kinematics, chosen swing joint, or precomputed trajectory. Five equal-time
# bins allow open-loop sequencing, while the remaining rows support feedback.
FEATURE_NAMES = (
    *(f"episode_time_phase_{index}" for index in range(N_TIME_PHASES)),
    *(f"joint{index}_gravity_dynamics_nm" for index in range(N_JOINTS)),
    *(f"joint{index}_angle_from_reset_rad" for index in range(N_JOINTS)),
    *(f"joint{index}_speed_rad_s" for index in range(N_JOINTS)),
    "ball_minus_club_face_x_m",
    "ball_minus_club_face_y_m",
    "ball_minus_club_face_z_m",
)
BIAS_START = N_TIME_PHASES
ANGLE_START = BIAS_START + N_JOINTS
SPEED_START = ANGLE_START + N_JOINTS
CLUB_TO_BALL_START = SPEED_START + N_JOINTS


def _default_matrix() -> np.ndarray:
    """Return a safe task-agnostic baseline, not a golf solution.

    It only compensates model dynamics and damps velocity. ProPS must add all
    positioning, club-face orientation, sequencing, and swing behavior.
    """
    matrix = np.zeros((len(FEATURE_NAMES), N_JOINTS), dtype=np.float64)
    damping = (-3.0, -3.0, -3.0, -1.5, -1.5, -1.5)
    for joint in range(N_JOINTS):
        matrix[BIAS_START + joint, joint] = 1.0
        matrix[SPEED_START + joint, joint] = damping[joint]
    return matrix


DEFAULT_MATRIX = _default_matrix()


def positive_shot_reward(distance_m: float, ball_hit: bool,
                         impact_centrality: float, ball_sunk: bool,
                         launch_speed_m_s: float = 0.0,
                         launch_direction_quality: float = 0.0) -> dict:
    """Return the terminal positive components (maximum 140 points)."""
    distance = float(distance_m)
    centrality = float(impact_centrality)
    launch_speed = float(launch_speed_m_s)
    direction = float(launch_direction_quality)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("distance_m must be finite and nonnegative")
    if not math.isfinite(centrality) or not 0.0 <= centrality <= 1.0:
        raise ValueError("impact_centrality must be between zero and one")
    if not math.isfinite(launch_speed) or launch_speed < 0:
        raise ValueError("launch_speed_m_s must be finite and nonnegative")
    if not math.isfinite(direction) or not 0.0 <= direction <= 1.0:
        raise ValueError("launch_direction_quality must be between zero and one")
    hit = bool(ball_hit)
    return {
        "contact": CONTACT_REWARD if hit else 0.0,
        "launch_speed": (
            LAUNCH_SPEED_REWARD
            * min(1.0, launch_speed / TARGET_LAUNCH_SPEED_M_S)
            if hit else 0.0
        ),
        "launch_direction": LAUNCH_DIRECTION_REWARD * direction if hit else 0.0,
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
        return np.clip(
            features @ self.matrix, -TORQUE_LIMITS_NM, TORQUE_LIMITS_NM
        ).astype(np.float32)


class GolfRLEnv(gym.Env):
    """Deterministic six-joint golf shot using Gymnasium's current API.

    Every episode begins from the same neutral pose. The controller gets no
    target joint angles or privileged trajectory. Shot quality is terminal;
    smoothness, collisions, and a small club-to-ball approach signal are
    incremental.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, duration_s: float = 5.0, frame_skip: int = 20,
                 render_mode: str | None = None, camera: str = "overview"):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
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
        angles = np.asarray(observation["joint_angles_rad"])
        speeds = np.asarray(observation["joint_speeds_rad_s"])
        features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        phase = min(N_TIME_PHASES - 1,
                    int((t / self.duration_s) * N_TIME_PHASES))
        features[phase] = 1.0
        for joint, name in enumerate(ACTION_JOINT_NAMES):
            dof = int(self.sim.model.joint(name).dofadr[0])
            features[BIAS_START + joint] = float(self.sim.data.qfrc_bias[dof])
            features[ANGLE_START + joint] = float(angles[joint])
            features[SPEED_START + joint] = float(speeds[joint])
        features[CLUB_TO_BALL_START:] = np.asarray(
            observation["club_to_ball_vector_m"], dtype=np.float32
        )
        return features

    def _reward_components(self) -> dict:
        positive = self._terminal_components.copy()
        return {
            **positive,
            "proximity": self._proximity_reward,
            "smoothness_penalty": -self._smoothness_penalty,
            "effort_penalty": -self._effort_penalty,
            "collision_penalty": -self._collision_penalty,
            "total": (
                sum(positive.values()) + self._proximity_reward
                - self._smoothness_penalty - self._effort_penalty
                - self._collision_penalty
            ),
        }

    def _info(self) -> dict:
        observation = self.sim.observe()
        return {
            **observation,
            "ball_settled": self._settled_after_hit_s >= BALL_SETTLE_TIME_S,
            "reward_components": self._reward_components(),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        observation = self.sim.reset()
        self._previous_action = np.zeros(N_JOINTS, dtype=np.float64)
        self._smoothness_penalty = 0.0
        self._effort_penalty = 0.0
        self._collision_penalty = 0.0
        self._proximity_reward = 0.0
        self._initial_club_distance = max(
            observation["closest_club_head_distance_m"], 1e-9
        )
        self._last_min_club_distance = observation["closest_club_head_distance_m"]
        self._last_collision_events = 0
        self._last_collision_time_s = 0.0
        self._settled_after_hit_s = 0.0
        self._terminal_components = {
            "contact": 0.0,
            "launch_speed": 0.0,
            "launch_direction": 0.0,
            "central_hit": 0.0,
            "distance": 0.0,
            "sunk": 0.0,
        }
        self._done = False
        return self._features(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("episode ended; call reset() before step()")
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (N_JOINTS,) or not np.all(np.isfinite(values)):
            raise ValueError("action must contain six finite joint torques")
        values = np.clip(values, -TORQUE_LIMITS_NM, TORQUE_LIMITS_NM)

        normalized_delta = (values - self._previous_action) / (2 * TORQUE_LIMITS_NM)
        smoothness_delta = SMOOTHNESS_PENALTY_WEIGHT * float(
            normalized_delta @ normalized_delta
        )
        self._smoothness_penalty += smoothness_delta
        normalized_action = values / TORQUE_LIMITS_NM
        effort_delta = EFFORT_PENALTY_WEIGHT * float(
            normalized_action @ normalized_action
        )
        self._effort_penalty += effort_delta
        self._previous_action = values.copy()

        for _ in range(self.frame_skip):
            observation = self.sim.step(values)
            if (observation["ball_hit"]
                    and observation["ball_speed_m_s"] <= BALL_SETTLE_SPEED_M_S):
                self._settled_after_hit_s += float(self.sim.model.opt.timestep)
            else:
                self._settled_after_hit_s = 0.0
            if (observation["time_s"] >= self.duration_s
                    or observation["ball_sunk"]
                    or self._settled_after_hit_s >= BALL_SETTLE_TIME_S):
                break

        new_min = observation["closest_club_head_distance_m"]
        raw_proximity_delta = PROXIMITY_REWARD_MAX * max(
            0.0, self._last_min_club_distance - new_min
        ) / self._initial_club_distance
        new_proximity_total = min(
            PROXIMITY_REWARD_MAX,
            self._proximity_reward + raw_proximity_delta,
        )
        proximity_delta = new_proximity_total - self._proximity_reward
        self._proximity_reward = new_proximity_total
        self._last_min_club_distance = new_min

        new_events = self.sim.collision_events - self._last_collision_events
        new_collision_time = (
            self.sim.collision_time_s - self._last_collision_time_s
        )
        raw_collision_delta = (
            COLLISION_EVENT_PENALTY * new_events
            + COLLISION_TIME_PENALTY_PER_S * new_collision_time
        )
        new_collision_total = min(
            COLLISION_PENALTY_CAP,
            self._collision_penalty + raw_collision_delta,
        )
        collision_delta = new_collision_total - self._collision_penalty
        self._collision_penalty = new_collision_total
        self._last_collision_events = self.sim.collision_events
        self._last_collision_time_s = self.sim.collision_time_s

        info = self._info()
        terminated = bool(info["ball_sunk"] or info["ball_settled"])
        truncated = not terminated and info["time_s"] >= self.duration_s
        self._done = terminated or truncated
        reward = (
            proximity_delta - smoothness_delta - effort_delta - collision_delta
        )
        if self._done:
            self._terminal_components = positive_shot_reward(
                info["distance_to_hole_m"], info["ball_hit"],
                info["impact_centrality"], info["ball_sunk"],
                info["launch_speed_m_s"], info["launch_direction_quality"],
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


gym.register(id="GolfSwing-v0", entry_point="golf_rl_env:GolfRLEnv")
