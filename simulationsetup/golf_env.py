"""MuJoCo golf environment with one torque-controlled wrist joint."""

import math
from pathlib import Path

import mujoco


SCENE = Path(__file__).resolve().parent / "golf_scene.xml"
HOLE_XY = (1.045, 0.0)
DEMO_PULSE_START = 0.05  # seconds


class GolfEnvironment:
    """Accept wrist motor torque and report ball-to-hole distance.

    `step(torque_nm, substeps)` advances the specified number of MuJoCo time
    steps. Positive torque drives the club toward the ball. Zero torque
    leaves the wrist free to move under inertia and gravity. The five other
    robot joints are braked at their address positions by equality constraints.
    """

    def __init__(self, scene_path=SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.ball_geom = self.model.geom("ball").id
        self.head_geom = self.model.geom("club_head").id
        self.min_distance_m = float("inf")
        self.hit = False
        self.reset()

    def reset(self):
        address = self.model.key("address").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, address)
        mujoco.mj_forward(self.model, self.data)
        self.hit = False
        self.min_distance_m = float("inf")
        return self.observe()

    def observe(self):
        ball = self.data.body("golf_ball").xpos
        dx = float(ball[0]) - HOLE_XY[0]
        dy = float(ball[1]) - HOLE_XY[1]
        distance = (dx * dx + dy * dy) ** 0.5
        self.min_distance_m = min(self.min_distance_m, distance)
        return {
            "time_s": float(self.data.time),
            "ball_position_m": (float(ball[0]), float(ball[1]), float(ball[2])),
            "distance_to_hole_m": distance,
            "closest_distance_m": self.min_distance_m,
            "swing_angle_rad": float(self.data.joint("joint5").qpos[0]),
            "wrist_angle_deg": math.degrees(
                float(self.data.joint("joint5").qpos[0]) - math.pi),
            "swing_speed_rad_s": float(self.data.joint("joint5").qvel[0]),
            "motor_torque_nm": float(self.data.ctrl[0]),
            "ball_hit": self.hit,
            "ball_sunk": distance < 0.045 and float(ball[2]) < -0.02,
        }

    def step(self, torque_nm, substeps=1):
        torque = float(torque_nm)
        if not math.isfinite(torque):
            raise ValueError("torque_nm must be finite")
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        self.data.ctrl[0] = torque
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
            self.observe()
            for contact in self.data.contact:
                if {contact.geom1, contact.geom2} == {self.ball_geom, self.head_geom}:
                    self.hit = True
        return self.observe()
