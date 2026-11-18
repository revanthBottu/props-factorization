"""Build the golf task from the supplied UR5 MuJoCo model.

All lengths are in metres.  The source robot model is left intact so it can
still be used on its own.  Run this file after changing ur5_arm_only.xml.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ur5_arm_only.xml"
OUTPUT = HERE / "golf_scene.xml"

BALL_RADIUS = 0.021335
BALL_MASS = 0.04593
SHAFT_LENGTH = 0.7096
SHAFT_OUTER_RADIUS = 0.017
HEAD_LENGTH = 0.127
HEAD_WIDTH = 0.0381
HEAD_HEIGHT = 0.0413
HEAD_HEEL_INSET = 0.012
HEAD_CENTER_X = HEAD_LENGTH / 2 - HEAD_HEEL_INSET
MOUNT_SCALE = 0.001  # STL is authored in millimetres
MOUNT_RADIUS = 0.0375
MOUNT_PIPE_START = 0.035
MOUNT_MASS = 0.12
PEDESTAL_BASE_WIDTH = 0.6064
PEDESTAL_BASE_HEIGHT = 0.10
PEDESTAL_COLUMN_WIDTH = 0.1524
PEDESTAL_COLUMN_HEIGHT = 0.70
HOLE_DISTANCE = 1.045
HOLE_WIDTH = 0.110

ARM_BASE_Z = PEDESTAL_BASE_HEIGHT + PEDESTAL_COLUMN_HEIGHT
# This is a simple task layout, not an inverse-kinematic solution. The robot
# starts from the same neutral all-zero joint state beside the ball. ProPS
# must discover both a useful pose and a swing through its policy matrix.
ROBOT_X = 0.0
ROBOT_Y = -0.55
UNIVERSAL_RESET_JOINTS = [0.0] * 6


def sub(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attributes.items()})


def indent(element, level=0):
    """Pretty-print XML on Python versions before ElementTree.indent."""
    prefix = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = prefix + "  "
        for child in element:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = prefix
    if level and (not element.tail or not element.tail.strip()):
        element.tail = prefix


def build():
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "ur5_golf")

    option = root.find("option")
    option.set("timestep", "0.001")
    option.set("gravity", "0 0 -9.81")

    asset = root.find("asset")
    sub(asset, "material", name="grass", rgba="0.22 0.43 0.21 1", specular="0.05")
    sub(asset, "material", name="pedestal", rgba="0.55 0.55 0.59 1")
    sub(asset, "material", name="club", rgba="0.75 0.76 0.78 1", specular="0.6")
    sub(asset, "material", name="pvc", rgba="0.86 0.87 0.89 1", specular="0.25")
    sub(asset, "material", name="mount", rgba="0.16 0.18 0.22 1", specular="0.25")
    sub(asset, "material", name="cup_liner", rgba="0.10 0.11 0.12 1")
    sub(asset, "mesh", name="club_mount_mesh", file="mount.STL",
        scale=f"{MOUNT_SCALE} {MOUNT_SCALE} {MOUNT_SCALE}")

    world = root.find("worldbody")
    robot = world.find("body[@name='base_link']")
    robot.set("pos", f"{ROBOT_X:.9f} {ROBOT_Y:.9f} {ARM_BASE_Z:.9f}")
    robot.set("euler", "0 0 0")
    # This scene's joint commands are physical angles.  The source model's
    # nonzero ref values only relabel those angles, so remove them here.
    for joint in robot.findall(".//joint"):
        joint.attrib.pop("ref", None)
    for index, damping in enumerate((0.20, 0.30, 0.30, 0.15, 0.10, 0.30)):
        robot.find(f".//joint[@name='joint{index}']").set("damping", str(damping))

    # Four green tiles leave a real opening.  Their top faces are at z=0.
    green = [
        ("green_west", -0.255, 0, 1.245, 1.30),
        ("green_east", 3.550, 0, 2.450, 1.30),
        ("green_north", HOLE_DISTANCE, 0.6775, 0.055, 0.6225),
        ("green_south", HOLE_DISTANCE, -0.6775, 0.055, 0.6225),
    ]
    for name, x, y, hx, hy in green:
        sub(world, "geom", name=name, type="box",
            pos=f"{x} {y} -0.04", size=f"{hx} {hy} 0.04",
            material="grass", friction="0.85 0.001 0.0012", condim="6")

    # The opening is 11 cm across.  The liner and bottom catch a sunk ball.
    sub(world, "geom", name="cup_bottom", type="box",
        pos=f"{HOLE_DISTANCE} 0 -0.13", size="0.055 0.055 0.005",
        material="cup_liner")
    for name, x, y, hx, hy in [
        ("cup_west", HOLE_DISTANCE - 0.0575, 0, 0.0025, 0.055),
        ("cup_east", HOLE_DISTANCE + 0.0575, 0, 0.0025, 0.055),
        ("cup_north", HOLE_DISTANCE, 0.0575, 0.055, 0.0025),
        ("cup_south", HOLE_DISTANCE, -0.0575, 0.055, 0.0025),
    ]:
        sub(world, "geom", name=name, type="box", pos=f"{x} {y} -0.065",
            size=f"{hx} {hy} 0.065", material="cup_liner")
    sub(world, "site", name="hole_target", pos=f"{HOLE_DISTANCE} 0 0",
        size="0.003", rgba="1 1 1 0")

    base = sub(world, "body", name="pedestal_base",
               pos=f"{ROBOT_X:.9f} {ROBOT_Y:.9f} 0")
    sub(base, "geom", name="pedestal_base_geom", type="box", pos="0 0 0.05",
        size="0.3032 0.3032 0.05", material="pedestal")
    sub(base, "geom", name="pedestal_column_geom", type="box",
        pos="0 0 0.45", size="0.0762 0.0762 0.35", material="pedestal")

    ball = sub(world, "body", name="golf_ball", pos=f"0 0 {BALL_RADIUS}")
    sub(ball, "freejoint", name="ball_freejoint")
    sub(ball, "geom", name="ball", type="sphere", size=BALL_RADIUS,
        mass=BALL_MASS, rgba="0.98 0.98 0.93 1",
        friction="0.55 0.001 0.0012", condim="6", solref="0.005 1")

    # Keep the source EE rotation: its +Z axis is collinear with the outward
    # axis of link6. The connector, PVC shaft, and last wrist link are thereby
    # parallel and rigidly attached, rather than using a task-specific angle.
    ee = robot.find(".//body[@name='EE']")
    club = sub(ee, "body", name="golf_club")
    sub(club, "geom", name="club_mount", type="mesh",
        mesh="club_mount_mesh", pos=f"{-MOUNT_RADIUS} {-MOUNT_RADIUS} 0",
        mass=MOUNT_MASS, material="mount",
        contype="0", conaffinity="0")
    sub(club, "geom", name="shaft", type="cylinder",
        fromto=f"0 0 {MOUNT_PIPE_START} 0 0 {MOUNT_PIPE_START + SHAFT_LENGTH}",
        size=SHAFT_OUTER_RADIUS, mass="0.22", material="pvc")
    head_z = MOUNT_PIPE_START + SHAFT_LENGTH + HEAD_HEIGHT / 2
    sub(club, "geom", name="club_head", type="box",
        pos=f"{HEAD_CENTER_X} 0 {head_z}",
        size=f"{HEAD_LENGTH / 2} {HEAD_WIDTH / 2} {HEAD_HEIGHT / 2}",
        mass="0.25", material="club", friction="0.6 0.04 0.01")
    sub(club, "site", name="club_face_center",
        pos=f"{HEAD_CENTER_X} {-HEAD_WIDTH / 2} {head_z}",
        size="0.002", rgba="1 0.8 0 1")

    sub(world, "light", name="sun", pos="0 0 4", dir="0 0 -1",
        diffuse="0.9 0.9 0.9")
    sub(world, "camera", name="overview", pos="-2.5 -4.2 3.2",
        xyaxes="0.86 -0.51 0 0.28 0.47 0.84", fovy="55")
    sub(world, "camera", name="impact", pos="-0.6 0.75 0.45",
        xyaxes="-0.781 -0.625 0 0.239 -0.299 0.921", fovy="45")
    sub(world, "camera", name="swing", pos="-0.85 0.65 1.15",
        xyaxes="-0.707 -0.707 0 0.315 -0.315 0.894", fovy="70")

    # All six joints remain free and torque controlled. No address or swing
    # keyframe is supplied to the controller.
    actuators = root.find("actuator")
    actuators.clear()
    for index in range(6):
        sub(actuators, "motor", name=f"joint{index}_torque",
            joint=f"joint{index}", gear="1")

    keyframe = sub(root, "keyframe")
    ball_qpos = [0, 0, BALL_RADIUS, 1, 0, 0, 0]
    ctrl = [0] * 6
    sub(keyframe, "key", name="universal",
        qpos=" ".join(f"{v:.9f}" for v in UNIVERSAL_RESET_JOINTS + ball_qpos),
        ctrl=" ".join(f"{v:.9f}" for v in ctrl))

    indent(root)
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {OUTPUT}")
    print(f"Pedestal centre: ({ROBOT_X:.4f}, {ROBOT_Y:.4f}) m")


if __name__ == "__main__":
    build()
