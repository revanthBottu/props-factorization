"""One-joint demo swing with an active wrist brake after contact.

The golf environment itself accepts raw torque. This controller supplies
the example torque pulse, then a damped hold command for the demo only.
"""

from golf_env import DEMO_PULSE_START


FOLLOW_THROUGH_RAD = 0.15
HOLD_KP = 40.0                 # N m / rad
HOLD_KD = 8.0                  # N m s / rad
WRIST_TORQUE_LIMIT = 28.0      # N m, from the supplied UR5 URDF


class DemoSwingController:
    def __init__(self, swing_torque=1.0, pulse_time=0.15):
        self.swing_torque = swing_torque
        self.pulse_time = pulse_time
        self.hold_angle = None

    def command(self, env):
        data = env.data
        joint = data.joint("joint5")

        if self.hold_angle is None and (env.hit or data.time >= 1.0):
            # Let the club follow through a little after ball contact, then
            # use velocity damping and position feedback to hold it there.
            self.hold_angle = float(joint.qpos[0]) + (
                FOLLOW_THROUGH_RAD if env.hit else 0.0)

        if self.hold_angle is None:
            return (self.swing_torque if DEMO_PULSE_START <= data.time
                    < DEMO_PULSE_START + self.pulse_time else 0.0)

        dof = env.model.joint("joint5").dofadr[0]
        angle_error = self.hold_angle - float(joint.qpos[0])
        angular_speed = float(joint.qvel[0])
        gravity_and_coriolis = float(data.qfrc_bias[dof])
        torque = (gravity_and_coriolis + HOLD_KP * angle_error
                  - HOLD_KD * angular_speed)
        return max(-WRIST_TORQUE_LIMIT, min(WRIST_TORQUE_LIMIT, torque))
