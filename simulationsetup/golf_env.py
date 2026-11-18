"""MuJoCo golf simulation with all six UR5 joints torque controlled."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


SCENE = Path(__file__).resolve().parent / "golf_scene.xml"
HOLE_XY = (1.045, 0.0)
BALL_RADIUS_M = 0.021335
ACTION_JOINT_NAMES = tuple(f"joint{index}" for index in range(6))
TARGET_DIRECTION_XY = np.array(HOLE_XY, dtype=np.float64)
TARGET_DIRECTION_XY /= np.linalg.norm(TARGET_DIRECTION_XY)

# Club-head dimensions used to normalize the first impact location. The
# impact offset is measured in the face plane, from the face-center site.
HEAD_HALF_LENGTH_M = 0.0635
HEAD_HALF_HEIGHT_M = 0.02065
MAX_FACE_OFFSET_M = math.hypot(HEAD_HALF_LENGTH_M, HEAD_HALF_HEIGHT_M)


class GolfEnvironment:
    """Accept six joint torques and expose deterministic shot diagnostics.

    Reset always restores the exact same neutral robot and ball state. No
    target pose, swing joint, or valid-strike trajectory is built into the
    environment; the policy controls every joint directly.
    """

    def __init__(self, scene_path=SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.ball_geom = self.model.geom("ball").id
        self.head_geom = self.model.geom("club_head").id
        self.face_center_local = np.array(
            self.model.site("club_face_center").pos, dtype=np.float64
        )
        self.robot_body_ids = self._descendant_body_ids(
            self.model.body("base_link").id
        )
        self.robot_geom_ids = {
            geom_id
            for geom_id, body_id in enumerate(self.model.geom_bodyid)
            if int(body_id) in self.robot_body_ids
        }
        self.reset()

    def _descendant_body_ids(self, root_id: int) -> set[int]:
        descendants = set()
        for body_id in range(self.model.nbody):
            cursor = body_id
            while cursor:
                if cursor == root_id:
                    descendants.add(body_id)
                    break
                cursor = int(self.model.body_parentid[cursor])
        return descendants

    def reset(self):
        universal = self.model.key("universal").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, universal)
        mujoco.mj_forward(self.model, self.data)
        self.hit = False
        self.min_distance_m = float("inf")
        self.min_club_head_distance_m = float("inf")
        self.impact_offset_m: float | None = None
        self.impact_centrality = 0.0
        self.collision_steps = 0
        self.collision_contacts = 0
        self.collision_events = 0
        self.collision_time_s = 0.0
        self._active_collision_pairs: set[str] = set()
        self.last_collision_pairs: tuple[str, ...] = ()
        self.collision_pair_events: dict[str, int] = {}
        self.max_ball_speed_m_s = 0.0
        self.launch_direction_quality = 0.0
        return self.observe()

    def _record_contacts(self) -> None:
        unwanted_this_step = 0
        unwanted_pairs: set[str] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair == {self.ball_geom, self.head_geom}:
                if not self.hit:
                    rotation = np.asarray(
                        self.data.body("golf_club").xmat, dtype=np.float64
                    ).reshape(3, 3)
                    local_contact = rotation.T @ (
                        np.asarray(contact.pos) - self.data.body("golf_club").xpos
                    )
                    dx = float(local_contact[0] - self.face_center_local[0])
                    dz = float(local_contact[2] - self.face_center_local[2])
                    self.impact_offset_m = math.hypot(dx, dz)
                    self.impact_centrality = float(np.clip(
                        1.0 - self.impact_offset_m / MAX_FACE_OFFSET_M, 0.0, 1.0
                    ))
                self.hit = True
            elif pair & self.robot_geom_ids:
                unwanted_this_step += 1
                names = sorted(
                    self.model.geom(geom_id).name or f"geom_{geom_id}"
                    for geom_id in pair
                )
                unwanted_pairs.add(" <-> ".join(names))

        new_pairs = unwanted_pairs - self._active_collision_pairs
        for pair_name in new_pairs:
            self.collision_pair_events[pair_name] = (
                self.collision_pair_events.get(pair_name, 0) + 1
            )
        self.collision_events += len(new_pairs)
        self._active_collision_pairs = unwanted_pairs
        self.last_collision_pairs = tuple(sorted(unwanted_pairs))
        if unwanted_pairs:
            self.collision_steps += 1
            self.collision_contacts += unwanted_this_step
            self.collision_time_s += float(self.model.opt.timestep)

    def _update_ball_motion(self) -> None:
        """Track the strongest horizontal launch produced after first impact."""
        if not self.hit:
            return
        velocity_xy = np.asarray(
            self.data.joint("ball_freejoint").qvel[:2], dtype=np.float64
        )
        speed = float(np.linalg.norm(velocity_xy))
        if speed > self.max_ball_speed_m_s:
            self.max_ball_speed_m_s = speed
            self.launch_direction_quality = (
                max(0.0, float(velocity_xy @ TARGET_DIRECTION_XY) / speed)
                if speed > 1e-9 else 0.0
            )

    def observe(self):
        ball = self.data.body("golf_ball").xpos
        dx = float(ball[0]) - HOLE_XY[0]
        dy = float(ball[1]) - HOLE_XY[1]
        distance = math.hypot(dx, dy)
        self.min_distance_m = min(self.min_distance_m, distance)
        face = np.asarray(self.data.site("club_face_center").xpos)
        club_to_ball = np.asarray(ball) - face
        club_head_distance = max(
            0.0, float(np.linalg.norm(club_to_ball)) - BALL_RADIUS_M
        )
        self.min_club_head_distance_m = min(
            self.min_club_head_distance_m, club_head_distance
        )
        joint_angles = np.array([
            float(self.data.joint(name).qpos[0]) for name in ACTION_JOINT_NAMES
        ])
        joint_speeds = np.array([
            float(self.data.joint(name).qvel[0]) for name in ACTION_JOINT_NAMES
        ])
        ball_velocity = np.asarray(
            self.data.joint("ball_freejoint").qvel[:3], dtype=np.float64
        )
        return {
            "time_s": float(self.data.time),
            "ball_position_m": (float(ball[0]), float(ball[1]), float(ball[2])),
            "distance_to_hole_m": distance,
            "closest_distance_m": self.min_distance_m,
            "club_head_distance_to_ball_m": club_head_distance,
            "closest_club_head_distance_m": self.min_club_head_distance_m,
            "club_to_ball_vector_m": tuple(float(value) for value in club_to_ball),
            "joint_angles_rad": tuple(float(value) for value in joint_angles),
            "joint_speeds_rad_s": tuple(float(value) for value in joint_speeds),
            "action_torques_nm": tuple(float(value) for value in self.data.ctrl),
            "ball_hit": self.hit,
            "impact_offset_m": self.impact_offset_m,
            "impact_centrality": self.impact_centrality,
            "ball_speed_m_s": float(np.linalg.norm(ball_velocity)),
            "launch_speed_m_s": self.max_ball_speed_m_s,
            "launch_direction_quality": self.launch_direction_quality,
            "collision_steps": self.collision_steps,
            "collision_contacts": self.collision_contacts,
            "collision_events": self.collision_events,
            "collision_time_s": self.collision_time_s,
            "active_collision_pairs": self.last_collision_pairs,
            "collision_pair_events": dict(self.collision_pair_events),
            "ball_sunk": distance < 0.045 and float(ball[2]) < -0.02,
        }

    def step(self, torques_nm: Sequence[float], substeps: int = 1):
        torques = np.asarray(torques_nm, dtype=np.float64)
        if torques.shape != (6,) or not np.all(np.isfinite(torques)):
            raise ValueError(
                "torques_nm must contain one finite torque for each of six joints"
            )
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        self.data.ctrl[:] = torques
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
            self._record_contacts()
            self._update_ball_motion()
            self.observe()
        return self.observe()
