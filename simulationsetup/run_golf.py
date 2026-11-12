"""Demo: swing the club and report the ball's distance from the hole.

The reusable action and observation API is in golf_env.py.  This script
chooses one example wrist torque pulse; a controller can choose its own later.
"""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import time

VIDEO_FPS = 60
VIDEO_DURATION = 2.5


def run(headless=False, swing_torque=1.0, pulse_time=0.15, duration=5.0,
        video_path=None, viewer_mode="auto", host="127.0.0.1", port=8765):
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    browser = not headless and video_path is None and (
        viewer_mode == "browser" or (viewer_mode == "auto" and not has_display))
    if browser:
        # Browser frames use offscreen OpenGL; select it before importing MuJoCo.
        if os.environ.get("MUJOCO_GL") not in ("egl", "osmesa"):
            os.environ["MUJOCO_GL"] = "osmesa"
        from browser_viewer import serve_demo
        print("No X11 display is needed for the browser view.")
        return serve_demo(swing_torque, pulse_time, duration, host, port)
    if not headless and video_path is None and not has_display:
        raise RuntimeError("No X11 display is available. Use --viewer browser or --video FILE.mp4.")

    # MuJoCo selects its OpenGL backend on import, so set this first.
    if video_path is not None and os.environ.get("MUJOCO_GL") not in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = "osmesa"

    import mujoco
    from demo_controller import DemoSwingController
    from golf_env import DEMO_PULSE_START, GolfEnvironment

    env = GolfEnvironment()
    controller = DemoSwingController(swing_torque, pulse_time)
    model, data = env.model, env.data
    print(f"Demo wrist torque: {swing_torque:+.2f} N m for {pulse_time:.2f} s "
          f"starting at {DEMO_PULSE_START:.2f} s; active brake after contact")

    viewer_context = nullcontext(None)
    if not headless and video_path is None:
        from mujoco import viewer as mjviewer
        viewer_context = mjviewer.launch_passive(model, data)

    renderer = writer = None
    if video_path is not None:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "Video recording needs imageio and imageio-ffmpeg: "
                "python -m pip install imageio imageio-ffmpeg") from exc
        video_path = Path(video_path).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=480, width=640)
        writer = imageio.get_writer(str(video_path), fps=VIDEO_FPS,
                                    codec="libx264", quality=8)

    try:
        with viewer_context as viewer:
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = model.camera("overview").id
                viewer.sync()
            next_frame = 0
            last_printed_second = -1
            while data.time < duration and (viewer is None or viewer.is_running()):
                step_start = time.perf_counter()
                observation = env.step(controller.command(env))
                second = int(observation["time_s"])
                if second > last_printed_second:
                    print(f"t={observation['time_s']:.1f}s  "
                          f"distance={observation['distance_to_hole_m']:.3f}m  "
                          f"wrist={observation['wrist_angle_deg']:+.1f}deg  "
                          f"hit={observation['ball_hit']}")
                    last_printed_second = second
                if (writer is not None and data.time <= VIDEO_DURATION
                        and data.time >= next_frame / VIDEO_FPS):
                    renderer.update_scene(data, camera="swing")
                    writer.append_data(renderer.render())
                    next_frame += 1
                if viewer is not None:
                    viewer.sync()
                    remaining = model.opt.timestep - (time.perf_counter() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    if video_path is not None:
        print(f"Demo video: {video_path}")

    result = env.observe()
    x, y, z = result["ball_position_m"]
    print(f"Ball final position: x={x:.3f} m, y={y:.3f} m, z={z:.3f} m")
    print(f"Distance from cup centre: {result['distance_to_hole_m']:.3f} m; "
          f"sunk: {result['ball_sunk']}")
    print(f"Wrist final speed: {result['swing_speed_rad_s']:.4f} rad/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="simulate without opening the viewer")
    parser.add_argument("--video", type=Path, metavar="FILE.mp4",
                        help="record an MP4 instead of opening the viewer")
    parser.add_argument("--viewer", choices=("auto", "desktop", "browser"),
                        default="auto", help="display mode (default: browser without X11)")
    parser.add_argument("--host", default="127.0.0.1", help="browser viewer bind address")
    parser.add_argument("--port", type=int, default=8765, help="browser viewer port")
    parser.add_argument("--swing-torque", type=float, default=1.0,
                        help="signed wrist motor torque in N m")
    parser.add_argument("--pulse-time", type=float, default=0.15,
                        help="seconds to apply the demo torque")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="total simulation time in seconds")
    args = parser.parse_args()
    if args.pulse_time <= 0 or args.duration <= 0:
        parser.error("--pulse-time and --duration must be positive")
    run(args.headless, args.swing_torque, args.pulse_time, args.duration,
        args.video, args.viewer, args.host, args.port)
