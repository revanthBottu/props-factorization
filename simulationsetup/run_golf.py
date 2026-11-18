"""Preview or record a matrix-controlled six-joint golf episode.

The default matrix only stabilizes the neutral reset pose. Pass a learned
``best_matrix.json`` with ``--matrix`` to replay behavior discovered by ProPS.
"""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import time


def _configure_offscreen_gl():
    """Select a headless backend where MuJoCo supports one."""
    if os.name != "nt" and os.environ.get("MUJOCO_GL") not in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = "osmesa"


def _load_matrix(path):
    from golf_rl_env import DEFAULT_MATRIX
    if path is None:
        return DEFAULT_MATRIX.copy()
    from run_golf_rl import parse_matrix_json
    return parse_matrix_json(Path(path).read_text(encoding="utf-8"))


def _print_result(result):
    print(f"Reward: {result['reward']:.4f}")
    print(f"Club hit: {result['ball_hit']}; centrality: "
          f"{result['impact_centrality']:.3f}")
    print(f"Launch speed: {result['launch_speed_m_s']:.3f} m/s; "
          f"direction quality: {result['launch_direction_quality']:.3f}")
    print(f"Distance from cup centre: {result['distance_to_hole_m']:.3f} m; "
          f"sunk: {result['ball_sunk']}")
    print(f"Closest club-face approach: "
          f"{result['closest_club_head_distance_m']:.3f} m")


def run(headless=False, duration=5.0, video_path=None, viewer_mode="auto",
        host="127.0.0.1", port=8765, matrix_path=None):
    has_display = os.name == "nt" or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    browser = not headless and video_path is None and (
        viewer_mode == "browser" or (viewer_mode == "auto" and not has_display)
    )
    matrix = _load_matrix(matrix_path)
    if browser:
        _configure_offscreen_gl()
        from browser_viewer import serve_demo
        return serve_demo(matrix, duration, host, port)

    if video_path is not None:
        _configure_offscreen_gl()
        from golf_rl_env import GolfRLEnv
        from run_golf_rl import rollout
        env = GolfRLEnv(duration_s=duration)
        try:
            result = rollout(matrix, env, video_path=video_path,
                             camera="overview")
        finally:
            env.close()
        print(f"Episode video: {Path(video_path).resolve()}")
        _print_result(result)
        return result

    from golf_rl_env import GolfRLEnv, MatrixGolfPolicy
    env = GolfRLEnv(duration_s=duration)
    policy = MatrixGolfPolicy(matrix)
    state, _ = env.reset()
    viewer_context = nullcontext(None)
    if not headless:
        import mujoco
        from mujoco import viewer as mjviewer
        viewer_context = mjviewer.launch_passive(env.sim.model, env.sim.data)
    total_reward = 0.0
    try:
        with viewer_context as viewer:
            if viewer is not None:
                import mujoco
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = env.sim.model.camera("overview").id
            while viewer is None or viewer.is_running():
                started = time.perf_counter()
                state, reward, terminated, truncated, info = env.step(
                    policy.action(state)
                )
                total_reward += reward
                if viewer is not None:
                    viewer.sync()
                    remaining = env.frame_skip * env.sim.model.opt.timestep - (
                        time.perf_counter() - started
                    )
                    if remaining > 0:
                        time.sleep(remaining)
                if terminated or truncated:
                    break
    finally:
        env.close()
    result = {**info, "reward": total_reward}
    _print_result(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true",
                        help="simulate without opening a viewer")
    parser.add_argument("--video", type=Path, metavar="FILE.mp4",
                        help="record an MP4 instead of opening a viewer")
    parser.add_argument("--matrix", type=Path,
                        help="best_matrix.json from a ProPS run")
    parser.add_argument("--viewer", choices=("auto", "desktop", "browser"),
                        default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    run(args.headless, args.duration, args.video, args.viewer,
        args.host, args.port, args.matrix)
