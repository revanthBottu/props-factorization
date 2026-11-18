"""Run the six-joint golf task through the shared ProPS Gymnasium workflow.

The older standalone rollout helpers remain importable for analysis, while
the CLI uses the same YAML configuration, agent, and log layout as other ProPS
environments.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np

# Select the offscreen backend before MuJoCo is imported by golf_rl_env.
if (platform.system() == "Linux"
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        and os.environ.get("MUJOCO_GL") not in ("egl", "osmesa")):
    os.environ["MUJOCO_GL"] = "osmesa"

from golf_rl_env import (ACTION_NAMES, DEFAULT_MATRIX, FEATURE_NAMES,
                         GolfRLEnv, MatrixGolfPolicy)


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
            camera: str = "overview") -> dict:
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
                    "launch_speed_m_s": info["launch_speed_m_s"],
                    "launch_direction_quality": info["launch_direction_quality"],
                    "ball_sunk": info["ball_sunk"],
                    "ball_settled": info["ball_settled"],
                    "ball_position_m": info["ball_position_m"],
                    "club_head_distance_to_ball_m": info["club_head_distance_to_ball_m"],
                    "closest_club_head_distance_m": info["closest_club_head_distance_m"],
                    "club_to_ball_vector_m": info["club_to_ball_vector_m"],
                    "joint_angles_rad": info["joint_angles_rad"],
                    "joint_speeds_rad_s": info["joint_speeds_rad_s"],
                    "collision_steps": info["collision_steps"],
                    "collision_contacts": info["collision_contacts"],
                    "collision_events": info["collision_events"],
                    "collision_time_s": info["collision_time_s"],
                    "active_collision_pairs": info["active_collision_pairs"],
                    "collision_pair_events": info["collision_pair_events"],
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
        "You optimize a numeric linear policy for a deterministic six-joint "
        "MuJoCo golf task. This is regular ProPS: do not factorize the matrix. "
        f"Return only JSON {{'matrix': ...}} with exactly {rows} rows and "
        f"{columns} finite numbers per row. Action = state @ matrix, with "
        "six columns ordered joint0 through joint5 torque. Joints 0-2 clip at "
        "+/-150 N m and joints 3-5 at +/-28 N m. Every reset uses the same "
        "neutral all-zero joint pose. No address pose, inverse kinematics, "
        "chosen swing joint, or swing trajectory is provided. The rows contain "
        "five generic episode-time bins, six model-dynamics values, six joint "
        "angles, six joint speeds, and the XYZ club-face-to-ball displacement. "
        "The 30-point strike reward is split into +10 contact, up to +10 "
        "launch speed, and up to +10 launch direction toward the cup. Reward "
        "also gives up to +40 for face-center impact, "
        "up to +40 for low final cup distance, +30 for a sink, and only a very "
        "small capped approach signal. It subtracts action slew, sustained "
        "effort, and very large event/time-based penalties for every robot or "
        "club collision other than the intended club-head/ball contact. The "
        "rollout reports the exact colliding geometry pairs. Final distance is "
        "measured after the ball settles when possible. The initial policy "
        "only compensates dynamics and damps "
        "velocity; discover positioning, face orientation, and swing behavior "
        "by editing the complete matrix."
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
             title="Six-joint golf reward progress")
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
          camera: str = "overview", create_plots: bool = True) -> list[dict]:
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
                f"effort={components['effort_penalty']:.3f} "
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
    """Run golf through the same config, agent, and logs as other ProPS tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", "--episodes", type=int,
                        help="number of ProPS LLM training episodes (default: config value)")
    parser.add_argument("--matrix", type=Path,
                        help="JSON file with a 26x6 'matrix' field for warmup")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    import sys
    import yaml

    props_dir = HERE.parent / "props-factorization"
    config_path = props_dir / "configs" / "golf" / "golf_props.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.iterations is not None:
        config["num_episodes"] = args.iterations
    if args.model:
        config["llm_model_name"] = (args.model if "/" in args.model
                                    else f"asu/{args.model}")
    if args.output_dir:
        config["logdir"] = str(args.output_dir.resolve())
    if args.matrix:
        config["initial_matrix"] = parse_matrix_json(
            args.matrix.read_text(encoding="utf-8")
        ).tolist()

    if str(props_dir) not in sys.path:
        sys.path.insert(0, str(props_dir))
    from runner.llm_num_optim_runner import run_training_loop

    # ProPS uses config paths relative to its project directory.
    previous_dir = Path.cwd()
    try:
        os.chdir(props_dir)
        run_training_loop(**config)
    finally:
        os.chdir(previous_dir)


if __name__ == "__main__":
    main()
