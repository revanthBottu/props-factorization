"""Task-agnostic six-joint stabilizer retained for raw-environment users.

This controller does not contain an address pose, swing timing, or golf
solution. Training and replay use the matrix controller in ``golf_rl_env``.
"""

from golf_env import ACTION_JOINT_NAMES


TORQUE_LIMITS = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
DAMPING = (3.0, 3.0, 3.0, 1.5, 1.5, 1.5)


class StabilizingController:
    """Compensate modeled dynamics and damp velocity at any configuration."""

    def command(self, env):
        torques = []
        for index, name in enumerate(ACTION_JOINT_NAMES):
            dof = int(env.model.joint(name).dofadr[0])
            speed = float(env.data.joint(name).qvel[0])
            torque = float(env.data.qfrc_bias[dof]) - DAMPING[index] * speed
            limit = TORQUE_LIMITS[index]
            torques.append(max(-limit, min(limit, torque)))
        return tuple(torques)


# Backward-compatible name; it no longer performs a scripted swing.
DemoSwingController = StabilizingController
