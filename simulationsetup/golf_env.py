"""MuJoCo golf simulation with independently controlled positioning and swing joints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


SCENE = Path(__file__).resolve().parent / "golf_scene.xml"
HOLE_XY = (1.045, 0.0)
DEMO_PULSE_START = 0.55  # allow joint4 to position the club before joint5 swings
CLUB_ADDRESS_ANGLE_RAD = -math.pi / 2
ACTION_JOINT_NAMES = ("joint4", "joint5")

# Club-head dimensions used to normalize the first impact location. The
# impact offset is measured in the face plane, from the face-center site.
HEAD_HALF_LENGTH_M = 0.0635
HEAD_HALF_HEIGHT_M = 0.02065
MAX_FACE_OFFSET_M = math.hypot(HEAD_HALF_LENGTH_M, HEAD_HALF_HEIGHT_M)


class GolfEnvironment:
    """Accept two wrist torques and expose deterministic shot diagnostics.

    ``step((position_torque, swing_torque), substeps)`` advances MuJoCo. The
    first four UR5 joints are fixed. Joint4 starts at a universal zero-angle
    pose and must turn to ``CLUB_ADDRESS_ANGLE_RAD``; joint5 performs the
    swing. Reset always restores the exact same robot and ball state.
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
        address = self.model.key("address").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, address)
        mujoco.mj_forward(self.model, self.data)
        self.hit = False
        self.min_distance_m = float("inf")
        self.impact_offset_m: float | None = None
        self.impact_centrality = 0.0
        self.collision_steps = 0
        self.collision_contacts = 0
        return self.observe()

    def _record_contacts(self) -> None:
        unwanted_this_step = 0
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

        if unwanted_this_step:
            self.collision_steps += 1
            self.collision_contacts += unwanted_this_step

    def observe(self):
        ball = self.data.body("golf_ball").xpos
        dx = float(ball[0]) - HOLE_XY[0]
        dy = float(ball[1]) - HOLE_XY[1]
        distance = math.hypot(dx, dy)
        self.min_distance_m = min(self.min_distance_m, distance)
        position_angle = float(self.data.joint("joint4").qpos[0])
        swing_angle = float(self.data.joint("joint5").qpos[0])
        return {
            "time_s": float(self.data.time),
            "ball_position_m": (float(ball[0]), float(ball[1]), float(ball[2])),
            "distance_to_hole_m": distance,
            "closest_distance_m": self.min_distance_m,
            "position_angle_rad": position_angle,
            "position_error_rad": CLUB_ADDRESS_ANGLE_RAD - position_angle,
            "position_speed_rad_s": float(self.data.joint("joint4").qvel[0]),
            "swing_angle_rad": swing_angle,
            "wrist_angle_deg": math.degrees(swing_angle - math.pi),
            "swing_speed_rad_s": float(self.data.joint("joint5").qvel[0]),
            "position_torque_nm": float(self.data.ctrl[0]),
            "motor_torque_nm": float(self.data.ctrl[1]),
            "action_torques_nm": tuple(float(value) for value in self.data.ctrl),
            "ball_hit": self.hit,
            "impact_offset_m": self.impact_offset_m,
            "impact_centrality": self.impact_centrality,
            "collision_steps": self.collision_steps,
            "collision_contacts": self.collision_contacts,
            "ball_sunk": distance < 0.045 and float(ball[2]) < -0.02,
        }

    def step(self, torques_nm: Sequence[float], substeps: int = 1):
        torques = np.asarray(torques_nm, dtype=np.float64)
        if torques.shape != (2,) or not np.all(np.isfinite(torques)):
            raise ValueError(
                "torques_nm must contain finite (position, swing) torques"
            )
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        self.data.ctrl[:] = torques
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
            self._record_contacts()
            self.observe()
        return self.observe()
