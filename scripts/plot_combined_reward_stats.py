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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _normalize_header(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _compress_below_threshold(
    values: Sequence[float],
    threshold: Optional[float],
    scale: float,
) -> np.ndarray:
    """Compress values below a threshold while keeping values above it unchanged.

    Mapping used when compression is enabled:
        y' = t - s * log1p(t - y), for y < t
        y' = y, for y >= t

    This preserves ordering and greatly reduces extreme negative spikes.
    """
    arr = np.asarray(values, dtype=float)
    if threshold is None:
        return arr

    if scale <= 0:
        raise ValueError(f"compression scale must be > 0, got {scale}")

    out = np.array(arr, copy=True)
    mask = arr < threshold
    if np.any(mask):
        out[mask] = threshold - (scale * np.log1p(threshold - arr[mask]))
    return out


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


def plot_combined(
    episodes: Sequence[int],
    means: np.ndarray,
    stds: np.ndarray,
    output_png: Path,
    title: str,
    source_labels: Sequence[str],
    show_runs: bool,
    run_series: Sequence[Dict[int, float]],
    compress_below: Optional[float],
    compress_below_scale: float,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))

    means_plot = _compress_below_threshold(means, compress_below, compress_below_scale)
    lower_plot = _compress_below_threshold(means - stds, compress_below, compress_below_scale)
    upper_plot = _compress_below_threshold(means + stds, compress_below, compress_below_scale)

    if show_runs:
        for label, series in zip(source_labels, run_series):
            y = [series[ep] for ep in episodes if ep in series]
            x = [ep for ep in episodes if ep in series]
            if x:
                y = _compress_below_threshold(y, compress_below, compress_below_scale)
                # Keep per-run overlays visible but out of the legend.
                plt.plot(x, y, alpha=0.25, linewidth=1.5)

    plt.plot(episodes, means_plot, color="#1f77b4", linewidth=2.5, label="mean reward")
    plt.fill_between(
        episodes,
        lower_plot,
        upper_plot,
        color="#1f77b4",
        alpha=0.2,
        label="mean +/- 1 std",
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

    if compress_below is not None:
        plt.axhline(
            y=float(compress_below),
            color="#444444",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label=f"compression threshold ({compress_below:g})",
        )

    plt.xlabel("Episode")
    if compress_below is None:
        plt.ylabel("Reward")
    else:
        plt.ylabel(f"Reward (values below {compress_below:g} compressed)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    if compress_below is not None:
        plt.figtext(
            0.5,
            0.01,
            (
                f"Below-threshold compression active: y' = t - s*log1p(t - y), "
                f"t={compress_below:g}, s={compress_below_scale:g}"
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
        "--compress-below",
        type=float,
        default=None,
        help=(
            "Optional threshold to compress values below it (for example -100). "
            "Values above threshold stay unchanged."
        ),
    )
    parser.add_argument(
        "--compress-below-scale",
        type=float,
        default=25.0,
        help=(
            "Compression strength for --compress-below (default: 25.0). "
            "Higher means more downward spread for values below threshold."
        ),
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

    episodes, means, stds, sample_counts = aggregate_rewards(
        rewards_by_run, args.start, args.end, args.require_all
    )

    if not episodes:
        raise SystemExit(
            "No episodes available in the requested range. "
            "Try adjusting --start/--end or remove --require-all."
        )

    output_png = Path(args.output)
    plot_combined(
        episodes=episodes,
        means=means,
        stds=stds,
        output_png=output_png,
        title=args.title,
        source_labels=labels,
        show_runs=args.show_runs,
        run_series=rewards_by_run,
        compress_below=args.compress_below,
        compress_below_scale=args.compress_below_scale,
    )

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

    print(f"Episodes plotted: {episodes[0]}..{episodes[-1]} ({len(episodes)} episodes)")
    print(f"Best mean reward: episode {best_episode}, reward {best_reward:.6f}")
    if args.compress_below is not None:
        print(
            "Applied below-threshold compression: "
            f"threshold={args.compress_below:g}, scale={args.compress_below_scale:g}"
        )
    print(f"Saved combined plot: {output_png.resolve()}")
    if args.summary_csv:
        print(f"Saved summary csv: {Path(args.summary_csv).resolve()}")


if __name__ == "__main__":
    main()
