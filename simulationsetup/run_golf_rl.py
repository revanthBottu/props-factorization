"""Run regular ProPS matrix search on the deterministic two-joint golf task.

Every evaluated episode gets a reward plot and policy heatmap. Whenever an
episode sets a new best reward, its deterministic replay is saved as an MP4.
No matrix factorization is used.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Select the offscreen backend before MuJoCo is imported by golf_rl_env.
if (platform.system() == "Linux"
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        and os.environ.get("MUJOCO_GL") not in ("egl", "osmesa")):
    os.environ["MUJOCO_GL"] = "osmesa"

from golf_rl_env import (ACTION_NAMES, DEFAULT_MATRIX, FEATURE_NAMES,
                         GolfRLEnv, MatrixGolfPolicy, TORQUE_LIMITS_NM)


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
        return
    load_dotenv(path, override=False)


load_local_env()


def rollout(matrix, env: GolfRLEnv, *, video_path: Path | None = None,
            camera: str = "swing") -> dict:
    """Run one deterministic episode from the universal reset pose."""
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
                raise RuntimeError(
                    "Video recording requires mujoco, imageio and imageio-ffmpeg"
                ) from exc
            video_path = Path(video_path).expanduser().resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(env.sim.model, height=480, width=640)
            writer = imageio.get_writer(
                str(video_path), fps=VIDEO_FPS, codec="libx264", quality=8
            )
            renderer.update_scene(env.sim.data, camera=camera)
            writer.append_data(renderer.render())
            next_frame = 1

        while True:
            action = policy.action(state)
            state, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            if renderer is not None:
                while info["time_s"] >= next_frame / VIDEO_FPS:
                    renderer.update_scene(env.sim.data, camera=camera)
                    writer.append_data(renderer.render())
                    next_frame += 1
            if terminated or truncated:
                result = {
                    "matrix": policy.matrix.tolist(),
                    "reward": float(cumulative_reward),
                    "reward_components": info["reward_components"],
                    "distance_to_hole_m": info["distance_to_hole_m"],
                    "closest_distance_m": info["closest_distance_m"],
                    "ball_hit": info["ball_hit"],
                    "impact_offset_m": info["impact_offset_m"],
                    "impact_centrality": info["impact_centrality"],
                    "ball_sunk": info["ball_sunk"],
                    "ball_position_m": info["ball_position_m"],
                    "position_angle_rad": info["position_angle_rad"],
                    "position_error_rad": info["position_error_rad"],
                    "position_speed_rad_s": info["position_speed_rad_s"],
                    "wrist_angle_deg": info["wrist_angle_deg"],
                    "wrist_speed_rad_s": info["swing_speed_rad_s"],
                    "collision_steps": info["collision_steps"],
                    "collision_contacts": info["collision_contacts"],
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
    """Ask the LLM to edit the complete regular ProPS matrix."""
    best = max(history, key=lambda entry: entry["reward"])
    rows, columns = DEFAULT_MATRIX.shape
    system = (
        "You optimize a numeric linear policy for a deterministic two-joint "
        "MuJoCo golf task. This is regular ProPS: do not factorize the matrix. "
        f"Return only JSON {{'matrix': ...}} with exactly {rows} rows and "
        f"{columns} finite numbers per row. Action = state @ matrix, with "
        "columns [joint4 positioning torque, joint5 swing torque], each clipped "
        "to +/-28 N m. Every reset starts joint4 at the universal 0 rad pose; "
        "the club face must be positioned near -pi/2 rad before joint5 can hit. "
        "A useful policy therefore controls both columns, first positioning and "
        "then swinging. Reward includes +20 for any hit, up to +30 for striking "
        "the face center, up to +30 for low final cup distance, and +20 for a "
        "sink. It subtracts action-slew (jerk proxy) and unwanted robot-contact "
        "penalties. Prefer smooth feedback and moderate matrix changes. Use the "
        "reported reward_components to diagnose each proposal."
    )
    context = {
        "feature_names_in_row_order": FEATURE_NAMES,
        "action_columns_in_order": ACTION_NAMES,
        "matrix_shape": [rows, columns],
        "best_so_far": best,
        "recent_trials": history[-8:],
        "instruction": (
            "Return the next complete unfactorized matrix, including unchanged "
            "entries. Do not return L/U, QR, SVD, or any factor matrices."
        ),
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context)},
        ],
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


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plots require matplotlib") from exc
    return plt


def plot_reward_progress(history: list[dict], output_dir: Path,
                         episode: int) -> tuple[Path, Path]:
    """Save both the latest reward graph and an immutable episode snapshot."""
    plt = _pyplot()
    episodes = [entry["trial"] for entry in history]
    rewards = [entry["reward"] for entry in history]
    best_index = int(np.argmax(rewards))
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(episodes, rewards, "b-o", markersize=4, linewidth=2)
    axis.axhline(rewards[best_index], color="red", linestyle="--", alpha=0.8,
                 label=f"Best: {rewards[best_index]:.2f}")
    axis.scatter([episodes[best_index]], [rewards[best_index]], color="red",
                 s=80, zorder=5)
    axis.set(xlabel="ProPS episode", ylabel="Reward",
             title="Two-joint golf reward progress")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    latest = output_dir / "reward_progress.png"
    snapshot = output_dir / f"reward_progress_ep{episode:03d}.png"
    fig.savefig(latest, dpi=150, bbox_inches="tight")
    fig.savefig(snapshot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return latest, snapshot


def plot_policy_heatmap(matrix: np.ndarray, output_dir: Path,
                        episode: int) -> Path:
    """Save the complete state-by-action matrix used for this episode."""
    plt = _pyplot()
    matrix = np.asarray(matrix, dtype=np.float64)
    limit = max(1.0, float(np.max(np.abs(matrix))))
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="coolwarm", aspect="auto",
                        vmin=-limit, vmax=limit)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.2f}",
                      ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(ACTION_NAMES)), ACTION_NAMES, rotation=15,
                    ha="right")
    axis.set_yticks(range(len(FEATURE_NAMES)), FEATURE_NAMES)
    axis.set(xlabel="Action column", ylabel="State-feature row",
             title=f"Regular ProPS policy matrix - episode {episode}")
    fig.colorbar(image, ax=axis, label="Weight value")
    fig.tight_layout()
    path = output_dir / f"policy_heatmap_ep{episode:03d}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_run_files(history: list[dict], output_dir: Path) -> None:
    best = max(history, key=lambda entry: entry["reward"])
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "best_matrix.json").write_text(
        json.dumps({
            "matrix": best["matrix"],
            "reward": best["reward"],
            "reward_components": best["reward_components"],
            "distance_to_hole_m": best["distance_to_hole_m"],
            "trial": best["trial"],
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def train(iterations: int, initial_matrix=DEFAULT_MATRIX, *, model: str | None = None,
          output_dir: Path | None = None, client=None, duration_s: float = 5.0,
          record_video: bool = False, record_best_video: bool = True,
          camera: str = "swing", create_plots: bool = True) -> list[dict]:
    """Evaluate a baseline plus ``iterations`` LLM revisions."""
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if (record_video or record_best_video or create_plots) and output_dir is None:
        raise ValueError("output_dir is required for plots or video")
    model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    if iterations and client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "Set OPENAI_API_KEY in simulationsetup/.env or your shell"
            )
        client = make_llm_client()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    env = GolfRLEnv(duration_s=duration_s)
    history: list[dict] = []
    best_reward = -float("inf")
    try:
        matrix = MatrixGolfPolicy(initial_matrix).matrix
        for trial in range(iterations + 1):
            if trial:
                matrix = propose_matrix(client, model, history)
            all_video_path = (
                output_dir / "all_videos" / f"trial_{trial:03d}.mp4"
                if record_video else None
            )
            result = rollout(matrix, env, video_path=all_video_path, camera=camera)
            result["trial"] = trial

            if result["reward"] > best_reward:
                previous = best_reward
                best_reward = result["reward"]
                print(f"New best: {best_reward:.4f} (previous {previous:.4f})")
                if record_best_video:
                    best_path = (output_dir / "best_videos" /
                                 f"best_trial_{trial:03d}_reward_{best_reward:.3f}.mp4")
                    try:
                        recorded = rollout(matrix, env, video_path=best_path,
                                           camera=camera)
                        if not np.isclose(recorded["reward"], result["reward"],
                                          atol=1e-9):
                            raise RuntimeError(
                                "deterministic best-video replay reward changed"
                            )
                        result["best_video_path"] = str(best_path.resolve())
                        print(f"Best video: {best_path}")
                    except Exception as exc:
                        result["best_video_error"] = str(exc)
                        print(f"Warning: could not record best video: {exc}")

            history.append(result)
            components = result["reward_components"]
            print(
                f"trial={trial} reward={result['reward']:.4f} "
                f"hit={result['ball_hit']} centrality={result['impact_centrality']:.3f} "
                f"distance={result['distance_to_hole_m']:.3f}m "
                f"jerk={components['smoothness_penalty']:.3f} "
                f"collision={components['collision_penalty']:.3f}", flush=True
            )
            if output_dir is not None:
                if create_plots:
                    reward_plot, _ = plot_reward_progress(history, output_dir, trial)
                    heatmap = plot_policy_heatmap(matrix, output_dir, trial)
                    print(f"Reward graph: {reward_plot}")
                    print(f"Matrix heatmap: {heatmap}")
                _save_run_files(history, output_dir)
    finally:
        env.close()
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=0,
                        help="number of LLM matrix revisions after the baseline")
    parser.add_argument("--matrix", type=Path,
                        help="JSON file with an 11x2 'matrix' field")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--record-video", action="store_true",
                        help="also record every trial; new bests are always recorded")
    parser.add_argument("--no-best-video", action="store_true",
                        help="disable automatic MP4 recording for new bests")
    parser.add_argument("--no-plots", action="store_true",
                        help="disable per-episode reward graphs and heatmaps")
    parser.add_argument("--camera", choices=VIDEO_CAMERAS, default="swing")
    args = parser.parse_args()

    matrix = (parse_matrix_json(args.matrix.read_text(encoding="utf-8"))
              if args.matrix else DEFAULT_MATRIX)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = HERE / "rl_runs" / stamp
    print(f"Features ({len(FEATURE_NAMES)} matrix rows): {', '.join(FEATURE_NAMES)}")
    print(f"Actions (2 matrix columns): {', '.join(ACTION_NAMES)}")
    print(f"Torque limits: +/-{TORQUE_LIMITS_NM.tolist()} N m")
    print("Policy type: regular ProPS (one full matrix; factorization disabled)")
    print(f"Results: {output_dir}")
    history = train(
        args.iterations, matrix, model=args.model, output_dir=output_dir,
        duration_s=args.duration, record_video=args.record_video,
        record_best_video=not args.no_best_video, camera=args.camera,
        create_plots=not args.no_plots,
    )
    best = max(history, key=lambda entry: entry["reward"])
    print(f"Best trial: {best['trial']}; reward={best['reward']:.4f}; "
          f"distance={best['distance_to_hole_m']:.3f} m")


if __name__ == "__main__":
    main()
