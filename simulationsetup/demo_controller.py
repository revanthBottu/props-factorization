"""Two-joint demo: position the club, swing, then hold both wrists.

The golf environment itself accepts raw torque. This controller supplies
the example torque pulse, then a damped hold command for the demo only.
"""

from golf_env import CLUB_ADDRESS_ANGLE_RAD, DEMO_PULSE_START


FOLLOW_THROUGH_RAD = 0.15
HOLD_KP = 40.0                 # N m / rad
HOLD_KD = 8.0                  # N m s / rad
WRIST_TORQUE_LIMIT = 28.0      # N m, from the supplied UR5 URDF
POSITION_KP = 20.0
POSITION_KD = 6.0


class DemoSwingController:
    def __init__(self, swing_torque=1.0, pulse_time=0.15):
        self.swing_torque = swing_torque
        self.pulse_time = pulse_time
        self.hold_angle = None

    def command(self, env):
        data = env.data
        position_joint = data.joint("joint4")
        swing_joint = data.joint("joint5")

        position_dof = env.model.joint("joint4").dofadr[0]
        position_torque = (
            float(data.qfrc_bias[position_dof])
            + POSITION_KP * (CLUB_ADDRESS_ANGLE_RAD - float(position_joint.qpos[0]))
            - POSITION_KD * float(position_joint.qvel[0])
        )

        if self.hold_angle is None and (env.hit or data.time >= 1.0):
            # Let the club follow through a little after ball contact, then
            # use velocity damping and position feedback to hold it there.
            self.hold_angle = float(swing_joint.qpos[0]) + (
                FOLLOW_THROUGH_RAD if env.hit else 0.0)

        if self.hold_angle is None:
            swing_torque = (self.swing_torque if DEMO_PULSE_START <= data.time
                            < DEMO_PULSE_START + self.pulse_time else 0.0)
            return (self._clip(position_torque), swing_torque)

        dof = env.model.joint("joint5").dofadr[0]
        angle_error = self.hold_angle - float(swing_joint.qpos[0])
        angular_speed = float(swing_joint.qvel[0])
        gravity_and_coriolis = float(data.qfrc_bias[dof])
        swing_torque = (gravity_and_coriolis + HOLD_KP * angle_error
                        - HOLD_KD * angular_speed)
        return (self._clip(position_torque), self._clip(swing_torque))

    @staticmethod
    def _clip(torque):
        return max(-WRIST_TORQUE_LIMIT, min(WRIST_TORQUE_LIMIT, torque))
