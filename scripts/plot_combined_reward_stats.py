#!/usr/bin/env python3
"""Create a combined reward plot (mean +/- std) across multiple runs.

Inputs can be:
- Run directories that contain overall_log.txt
- Direct paths to overall_log.txt
- Any file/dir inside a run folder (the script walks up parents to find overall_log.txt)

Example:
  python scripts/plot_combined_reward_stats.py \
      logs/swimmer_run_a logs/swimmer_run_b logs/swimmer_run_c \
      --start 0 --end 113 \
      --output logs/combined_swimmer_0_113.png
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _normalize_header(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _get_window_color(
    value: float, lower_bound: float = -400.0, upper_bound: float = 1000.0, severity_scale: float = 600.0
) -> Tuple[float, float, float]:
    """Get RGB color based on distance from window [lower_bound, upper_bound].

    Returns blue for values within window, transitioning to red for values outside.
    Severity increases with distance from the window.
    """
    if lower_bound <= value <= upper_bound:
        return (0.12, 0.47, 0.71)  # Blue #1f77b4

    # Calculate distance outside the window
    if value < lower_bound:
        distance = lower_bound - value
    else:  # value > upper_bound
        distance = value - upper_bound

    # Normalize to 0-1 range
    severity = min(distance / severity_scale, 1.0)

    # Interpolate from blue to red
    blue_rgb = (0.12, 0.47, 0.71)  # Blue
    red_rgb = (0.84, 0.15, 0.16)   # Red #d62728

    r = blue_rgb[0] + (red_rgb[0] - blue_rgb[0]) * severity
    g = blue_rgb[1] + (red_rgb[1] - blue_rgb[1]) * severity
    b = blue_rgb[2] + (red_rgb[2] - blue_rgb[2]) * severity

    return (r, g, b)


def resolve_overall_log(input_path: str) -> Path:
    """Resolve a user-provided path to an overall_log.txt file."""
    path = Path(input_path).expanduser().resolve()

    if path.is_file() and path.name == "overall_log.txt":
        return path

    if path.is_dir():
        candidate = path / "overall_log.txt"
        if candidate.exists():
            return candidate

    # If a nested file/dir is provided (for example: episode_113/reward_progress.png),
    # climb ancestors until we find overall_log.txt.
    start = path if path.is_dir() else path.parent
    for parent in [start, *start.parents]:
        candidate = parent / "overall_log.txt"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find overall_log.txt for input: {input_path}")


def load_episode_rewards(overall_log: Path) -> Dict[int, float]:
    """Load zero-based episode -> total reward mapping from overall_log.txt."""
    rewards_by_episode: Dict[int, float] = {}

    with overall_log.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty log file: {overall_log}")

        normalized = [_normalize_header(h) for h in header]

        try:
            iteration_idx = normalized.index("iteration")
        except ValueError as exc:
            raise ValueError(f"Missing 'Iteration' column in: {overall_log}") from exc

        try:
            reward_idx = normalized.index("total reward")
        except ValueError as exc:
            raise ValueError(f"Missing 'Total Reward' column in: {overall_log}") from exc

        for row in reader:
            if not row:
                continue
            if max(iteration_idx, reward_idx) >= len(row):
                continue

            try:
                iteration = int(float(row[iteration_idx].strip()))
                reward = float(row[reward_idx].strip())
            except ValueError:
                continue

            # Iteration is 1-based in overall_log.txt, convert to zero-based episode id.
            episode = iteration - 1
            if episode >= 0:
                rewards_by_episode[episode] = reward

    if not rewards_by_episode:
        raise ValueError(f"No valid reward rows found in: {overall_log}")

    return rewards_by_episode


def aggregate_rewards(
    episode_rewards_by_run: Sequence[Dict[int, float]],
    start: int,
    end: int,
    require_all: bool,
) -> Tuple[List[int], np.ndarray, np.ndarray, List[int]]:
    episodes: List[int] = []
    means: List[float] = []
    stds: List[float] = []
    sample_counts: List[int] = []

    num_runs = len(episode_rewards_by_run)

    for episode in range(start, end + 1):
        values = [run[episode] for run in episode_rewards_by_run if episode in run]

        if require_all and len(values) != num_runs:
            continue
        if not values:
            continue

        arr = np.asarray(values, dtype=float)
        episodes.append(episode)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr)))
        sample_counts.append(len(values))

    return episodes, np.asarray(means), np.asarray(stds), sample_counts


def write_summary_csv(
    output_csv: Path,
    episodes: Sequence[int],
    means: np.ndarray,
    stds: np.ndarray,
    sample_counts: Sequence[int],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "mean_reward", "std_reward", "num_runs"])
        for ep, mean, std, n in zip(episodes, means, stds, sample_counts):
            writer.writerow([ep, mean, std, n])


def plot_raw_episodes(
    run_series: Sequence[Dict[int, float]],
    output_png: Path,
    title: str,
    source_labels: Sequence[str],
    lower_bound: float = -400.0,
    upper_bound: float = 1000.0,
) -> None:
    """Plot raw individual episode rewards with visual spike clipping."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 7))

    for label, series in zip(source_labels, run_series):
        episodes = sorted(series.keys())
        rewards = [series[ep] for ep in episodes]

        if not episodes:
            continue

        # Clip for display only (doesn't affect the actual data)
        rewards_clipped = np.clip(rewards, lower_bound, upper_bound)

        # Plot each segment with color based on whether it's outside the window
        for i in range(len(episodes) - 1):
            color = _get_window_color(rewards[i], lower_bound, upper_bound)
            plt.plot(
                episodes[i : i + 2],
                rewards_clipped[i : i + 2],
                color=color,
                alpha=0.6,
                linewidth=1.2,
                label=label if i == 0 else "",
            )

    plt.axhline(
        y=float(lower_bound),
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label=f"window bounds ({lower_bound:g} to {upper_bound:g})",
    )
    plt.axhline(
        y=float(upper_bound),
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.ylim(lower_bound - 100, upper_bound + 100)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    plt.figtext(
        0.5,
        0.01,
        (
            f"Individual episode rewards | "
            f"Blue→Red gradient indicates spike intensity outside [{lower_bound:g}, {upper_bound:g}] window | "
            f"Rewards beyond window are clipped visually for trend visibility"
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def plot_combined(
    episodes: Sequence[int],
    means: np.ndarray,
    stds: np.ndarray,
    output_png: Path,
    title: str,
    source_labels: Sequence[str],
    show_runs: bool,
    run_series: Sequence[Dict[int, float]],
    lower_bound: float = -400.0,
    upper_bound: float = 1000.0,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))

    # Clip values to window for display
    means_plot = np.clip(means, lower_bound, upper_bound)
    lower_plot = np.clip(means - stds, lower_bound, upper_bound)
    upper_plot = np.clip(means + stds, lower_bound, upper_bound)

    if show_runs:
        for label, series in zip(source_labels, run_series):
            y = [series[ep] for ep in episodes if ep in series]
            x = [ep for ep in episodes if ep in series]
            if x:
                y_clipped = np.clip(y, lower_bound, upper_bound)
                # Color segments based on distance from window
                for i in range(len(x) - 1):
                    color = _get_window_color(y[i], lower_bound, upper_bound)
                    plt.plot(
                        x[i : i + 2],
                        y_clipped[i : i + 2],
                        color=color,
                        alpha=0.25,
                        linewidth=1.5,
                    )

    # Plot lower bound (mean - std) with gradient coloring
    for i in range(len(episodes) - 1):
        color = _get_window_color(means[i] - stds[i], lower_bound, upper_bound)
        plt.plot(
            episodes[i : i + 2],
            lower_plot[i : i + 2],
            color=color,
            alpha=0.4,
            linewidth=1.0,
            linestyle="--",
        )

    # Plot mean with gradient coloring
    for i in range(len(episodes) - 1):
        color = _get_window_color(means[i], lower_bound, upper_bound)
        plt.plot(
            episodes[i : i + 2],
            means_plot[i : i + 2],
            color=color,
            linewidth=2.5,
        )

    # Plot upper bound (mean + std) with gradient coloring
    for i in range(len(episodes) - 1):
        color = _get_window_color(means[i] + stds[i], lower_bound, upper_bound)
        plt.plot(
            episodes[i : i + 2],
            upper_plot[i : i + 2],
            color=color,
            alpha=0.4,
            linewidth=1.0,
            linestyle="--",
        )

    # Highlight the best mean reward episode on the combined curve.
    best_idx = int(np.argmax(means))
    best_episode = int(episodes[best_idx])
    best_reward = float(means[best_idx])
    best_reward_plot = float(means_plot[best_idx])
    plt.scatter(
        [best_episode],
        [best_reward_plot],
        color="#d62728",
        s=90,
        zorder=5,
        label="best mean reward",
    )
    plt.annotate(
        f"Best: ep {best_episode}, reward {best_reward:.2f}",
        xy=(best_episode, best_reward_plot),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=10,
        color="#d62728",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1),
    )

    plt.axhline(
        y=float(lower_bound),
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label=f"window bounds ({lower_bound:g} to {upper_bound:g})",
    )
    plt.axhline(
        y=float(upper_bound),
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.ylim(lower_bound - 100, upper_bound + 100)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    plt.figtext(
        0.5,
        0.01,
        (
            f"Solid line: mean | Dashed lines: ±1 std | "
            f"Blue→Red gradient indicates spike intensity outside [{lower_bound:g}, {upper_bound:g}] window"
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine rewards across runs and plot episode-wise mean +/- std."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Run folders, overall_log.txt files, or nested paths within a run folder.",
    )
    parser.add_argument("--start", type=int, required=True, help="Start episode (inclusive).")
    parser.add_argument("--end", type=int, required=True, help="End episode (inclusive).")
    parser.add_argument(
        "--output",
        default="logs/combined_reward_mean_std.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional CSV with per-episode mean/std values.",
    )
    parser.add_argument(
        "--title",
        default="Combined Reward: Mean +/- Std",
        help="Plot title.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Only include episodes present in every run.",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Overlay each run as a faint line.",
    )
    parser.add_argument(
        "--lower-bound",
        type=float,
        default=-400.0,
        help="Lower bound of display window (default: -400).",
    )
    parser.add_argument(
        "--upper-bound",
        type=float,
        default=1000.0,
        help="Upper bound of display window (default: 1000).",
    )
    parser.add_argument(
        "--raw-episodes",
        action="store_true",
        help="Plot individual episode rewards instead of mean/std (good for seeing trends despite spikes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.end < args.start:
        raise SystemExit("--end must be >= --start")

    resolved_logs: List[Path] = []
    for raw in args.inputs:
        resolved_logs.append(resolve_overall_log(raw))

    rewards_by_run: List[Dict[int, float]] = []
    labels: List[str] = []
    skipped_logs: List[Tuple[Path, str]] = []
    for log_file in resolved_logs:
        try:
            rewards = load_episode_rewards(log_file)
        except Exception as exc:
            skipped_logs.append((log_file, str(exc)))
            continue
        rewards_by_run.append(rewards)
        labels.append(log_file.parent.name)

    if not rewards_by_run:
        raise SystemExit("No usable logs were found in the provided inputs.")

    output_png = Path(args.output)

    if args.raw_episodes:
        # Filter rewards to the requested range and plot raw episode data
        filtered_rewards = []
        for run in rewards_by_run:
            filtered_run = {ep: reward for ep, reward in run.items() if args.start <= ep <= args.end}
            if filtered_run:
                filtered_rewards.append(filtered_run)

        if not filtered_rewards:
            raise SystemExit(
                "No episodes available in the requested range. "
                "Try adjusting --start/--end or remove --require-all."
            )

        plot_raw_episodes(
            run_series=filtered_rewards,
            output_png=output_png,
            title=args.title,
            source_labels=labels,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
        )
        print(f"Plotted raw individual episode rewards: episodes {args.start}..{args.end}")
    else:
        episodes, means, stds, sample_counts = aggregate_rewards(
            rewards_by_run, args.start, args.end, args.require_all
        )

        if not episodes:
            raise SystemExit(
                "No episodes available in the requested range. "
                "Try adjusting --start/--end or remove --require-all."
            )

        plot_combined(
            episodes=episodes,
            means=means,
            stds=stds,
            output_png=output_png,
            title=args.title,
            source_labels=labels,
            show_runs=args.show_runs,
            run_series=rewards_by_run,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
        )

    if not args.raw_episodes:
        best_idx = int(np.argmax(means))
        best_episode = int(episodes[best_idx])
        best_reward = float(means[best_idx])

        if args.summary_csv:
            write_summary_csv(Path(args.summary_csv), episodes, means, stds, sample_counts)

    print(f"Resolved logs: {len(resolved_logs)}")
    for idx, p in enumerate(resolved_logs, start=1):
        print(f"  {idx}. {p}")

    if skipped_logs:
        print("Skipped logs:")
        for p, reason in skipped_logs:
            print(f"  - {p}: {reason}")

    if not args.raw_episodes:
        print(f"Episodes plotted: {episodes[0]}..{episodes[-1]} ({len(episodes)} episodes)")
        print(f"Best mean reward: episode {best_episode}, reward {best_reward:.6f}")
        print(
            f"Window bounds: [{args.lower_bound:g}, {args.upper_bound:g}] "
            f"(blue→red gradient for values outside window)"
        )
        if args.summary_csv:
            print(f"Saved summary csv: {Path(args.summary_csv).resolve()}")

    print(f"Saved plot: {output_png.resolve()}")


if __name__ == "__main__":
    main()
