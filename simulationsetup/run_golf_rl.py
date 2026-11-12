"""Evaluate and revise a ProPS-style matrix policy on the golf task.

No API call is made for the baseline rollout. Fill in OPENAI_API_KEY in .env,
then use --iterations N for revisions through the ASU Research Computing API.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# Select the offscreen backend before MuJoCo is imported by golf_rl_env.
if (not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        and os.environ.get("MUJOCO_GL") not in ("egl", "osmesa")):
    os.environ["MUJOCO_GL"] = "osmesa"

from golf_rl_env import (DEFAULT_MATRIX, FEATURE_NAMES, GolfRLEnv,
                         MatrixGolfPolicy, REWARD_OFFSET_M, TORQUE_LIMIT_NM)


HERE = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://openai.rc.asu.edu/v1"
DEFAULT_MODEL = "gemma4-31b-it"
VIDEO_FPS = 30
VIDEO_CAMERAS = ("swing", "impact", "overview")


def load_local_env(path: Path = HERE / ".env") -> None:
    """Read local credentials without replacing shell environment values."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # exported shell variables still work for API training
    load_dotenv(path, override=False)


load_local_env()


def rollout(matrix, env: GolfRLEnv, *, video_path: Path | None = None,
            camera: str = "swing") -> dict:
    """Run one deterministic shot from the address keyframe."""
    if camera not in VIDEO_CAMERAS:
        raise ValueError(f"camera must be one of {VIDEO_CAMERAS}")
    policy = MatrixGolfPolicy(matrix)
    state, _ = env.reset()
    cumulative_reward = 0.0
    renderer = writer = None
    try:
        if video_path is not None:
            try:
                import imageio.v2 as imageio
                import mujoco
            except ImportError as exc:
                raise RuntimeError("Video recording requires imageio and "
                                   "imageio-ffmpeg") from exc
            video_path = Path(video_path).expanduser().resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(env.sim.model, height=480, width=640)
            writer = imageio.get_writer(str(video_path), fps=VIDEO_FPS,
                                        codec="libx264", quality=8)
            renderer.update_scene(env.sim.data, camera=camera)
            writer.append_data(renderer.render())
            next_frame = 1
        while True:
            state, reward, terminated, truncated, info = env.step(policy.action(state))
            cumulative_reward += reward
            if renderer is not None and info["time_s"] >= next_frame / VIDEO_FPS:
                renderer.update_scene(env.sim.data, camera=camera)
                writer.append_data(renderer.render())
                next_frame += 1
            if terminated or truncated:
                result = {
                    "matrix": policy.matrix.tolist(),
                    "reward": cumulative_reward,
                    "distance_to_hole_m": info["distance_to_hole_m"],
                    "closest_distance_m": info["closest_distance_m"],
                    "ball_hit": info["ball_hit"],
                    "ball_sunk": info["ball_sunk"],
                    "ball_position_m": info["ball_position_m"],
                    "wrist_angle_deg": info["wrist_angle_deg"],
                    "wrist_speed_rad_s": info["wrist_speed_rad_s"],
                    "time_s": info["time_s"],
                }
                if video_path is not None:
                    result["video_path"] = str(video_path)
                return result
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()


def parse_matrix_json(content: str) -> np.ndarray:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    payload = json.loads(content)
    if not isinstance(payload, dict) or "matrix" not in payload:
        raise ValueError("LLM response must be a JSON object with a matrix field")
    return MatrixGolfPolicy(payload["matrix"]).matrix


def propose_matrix(client, model: str, history: list[dict]) -> np.ndarray:
    """Ask an API model to edit the 7x1 torque matrix using rollout results."""
    best = max(history, key=lambda entry: entry["reward"])
    system = (
        "You optimize a numeric linear policy for a deterministic MuJoCo golf task. "
        "Return only a JSON object with a 'matrix' field containing seven rows, "
        "one finite number per row. Change the matrix to improve terminal reward. "
        "The joint torque is observation @ matrix, clipped to +/-28 N m. "
        "Only the UR5 wrist joint moves; other robot joints are locked. "
        "The ball starts 1.045 m from the cup. Ground friction, ball mass, "
        "geometry, and initial pose are fixed. "
        "The only reward is 1/(final horizontal distance in metres + 0.02), "
        "so a shot that passes the cup then rolls away scores poorly. "
        "The maximum reward is 50 when final distance is zero. "
        "Before contact the first four observation entries are time-window indicators. "
        "After contact, or at the 1 s timeout, they are zero and the last three "
        "entries provide gravity-compensation torque in N m, target minus current "
        "wrist angle in radians, and wrist speed in rad/s. The target is the contact "
        "angle plus 0.15 rad of follow-through. The baseline hold weights are "
        "1, 40, -8 for these last three inputs. Choose moderate changes first."
    )
    context = {
        "feature_names_in_row_order": FEATURE_NAMES,
        "best_so_far": best,
        "recent_trials": history[-8:],
        "instruction": "Return the next complete 7x1 matrix, including unchanged rows.",
    }
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": json.dumps(context)}],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned no matrix")
    return parse_matrix_json(content)


def make_llm_client():
    """Create the ASU-compatible OpenAI client from local environment values."""
    from openai import OpenAI
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
        api_key=os.environ["OPENAI_API_KEY"],
    )


def train(iterations: int, initial_matrix=DEFAULT_MATRIX, *, model: str | None = None,
          output_dir: Path | None = None, client=None, duration_s: float = 5.0,
          record_video: bool = False, camera: str = "swing") -> list[dict]:
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if record_video and output_dir is None:
        raise ValueError("output_dir is required when record_video is enabled")
    model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    if iterations and client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "Set OPENAI_API_KEY in simulationsetup/.env or your shell "
                "(install python-dotenv to load .env)")
        client = make_llm_client()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    env = GolfRLEnv(duration_s=duration_s)
    history = []
    try:
        matrix = MatrixGolfPolicy(initial_matrix).matrix
        for trial in range(iterations + 1):
            if trial:
                matrix = propose_matrix(client, model, history)
            video_path = output_dir / f"trial_{trial:03d}.mp4" if record_video else None
            result = rollout(matrix, env, video_path=video_path, camera=camera)
            result["trial"] = trial
            history.append(result)
            print(f"trial={trial} reward={result['reward']:.4f} "
                  f"final_distance={result['distance_to_hole_m']:.3f}m "
                  f"hit={result['ball_hit']} sunk={result['ball_sunk']}", flush=True)
            if video_path is not None:
                print(f"Video: {video_path}", flush=True)
            if output_dir is not None:
                best = max(history, key=lambda entry: entry["reward"])
                (output_dir / "history.json").write_text(
                    json.dumps(history, indent=2) + "\n", encoding="utf-8")
                (output_dir / "best_matrix.json").write_text(
                    json.dumps({"matrix": best["matrix"], "reward": best["reward"],
                                "distance_to_hole_m": best["distance_to_hole_m"]},
                               indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=0,
                        help="number of LLM matrix revisions after the baseline")
    parser.add_argument("--matrix", type=Path,
                        help="JSON file with a 7x1 'matrix' field to evaluate first")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL,
                        help="API model name (or set OPENAI_MODEL)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="seconds per shot; default 5")
    parser.add_argument("--output-dir", type=Path,
                        help="directory for history.json and best_matrix.json")
    parser.add_argument("--record-video", action="store_true",
                        help="save one MP4 for each evaluated matrix")
    parser.add_argument("--camera", choices=VIDEO_CAMERAS, default="swing",
                        help="camera used for recorded MP4s")
    args = parser.parse_args()
    matrix = (parse_matrix_json(args.matrix.read_text(encoding="utf-8"))
              if args.matrix else DEFAULT_MATRIX)
    output_dir = args.output_dir
    if output_dir is None and (args.iterations or args.record_video):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = HERE / "rl_runs" / stamp
    print(f"Features (matrix rows): {', '.join(FEATURE_NAMES)}")
    print(f"Action: wrist torque, limited to +/-{TORQUE_LIMIT_NM:g} N m")
    print(f"Terminal reward: 1 / (distance_m + {REWARD_OFFSET_M:g})")
    if output_dir is not None:
        print(f"Results: {output_dir}")
    history = train(args.iterations, matrix, model=args.model,
                    output_dir=output_dir, duration_s=args.duration,
                    record_video=args.record_video, camera=args.camera)
    best = max(history, key=lambda entry: entry["reward"])
    print(f"Best trial: {best['trial']}; reward={best['reward']:.4f}; "
          f"distance={best['distance_to_hole_m']:.3f} m")


if __name__ == "__main__":
    main()
