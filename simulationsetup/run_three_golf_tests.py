"""Run sequential golf experiments using the golf YAML as source of truth.

The script changes only the seed and log directory for each independent run.
Episode count, model, policy dimensions, search settings, and every other
training parameter come directly from ``configs/golf/golf_props.yaml``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import yaml

from run_golf_rl import load_local_env


HERE = Path(__file__).resolve().parent
PROPS_DIR = HERE.parent / "props-factorization"
CONFIG_PATH = PROPS_DIR / "configs" / "golf" / "golf_props.yaml"


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _numbered_logdir(base: Path, number: int) -> Path:
    """Return ``base`` for zero, then ``base_1``, ``base_2``, and so on."""
    return base if number == 0 else base.with_name(f"{base.name}_{number}")


def _first_available_series(base: Path, count: int) -> tuple[int, list[Path]]:
    """Find a consecutive log-name series without overwriting prior runs."""
    start = 0
    while True:
        candidates = [_numbered_logdir(base, start + offset)
                      for offset in range(count)]
        if not any(path.exists() for path in candidates):
            return start, candidates
        start += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3,
                        help="number of sequential independent tests (default: 3)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    load_local_env()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required in simulationsetup/.env or the shell")

    base_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    episodes = int(base_config["num_episodes"])
    base_seed = int(base_config["seed"])
    configured_logdir = Path(str(base_config["logdir"]))
    if episodes < 1:
        parser.error("num_episodes in golf_props.yaml must be positive")
    if not configured_logdir.is_absolute():
        configured_logdir = PROPS_DIR / configured_logdir
    configured_logdir = configured_logdir.resolve()
    configured_logdir.parent.mkdir(parents=True, exist_ok=True)

    start_number, test_dirs = _first_available_series(
        configured_logdir, args.runs
    )
    manifest_suffix = "" if start_number == 0 else f"_{start_number}"
    manifest_path = configured_logdir.parent / (
        f"{configured_logdir.name}_batch_manifest{manifest_suffix}.json"
    )
    if manifest_path.exists():
        parser.error(f"batch manifest already exists: {manifest_path}")

    manifest = {
        "config": str(CONFIG_PATH),
        "yaml_logdir": str(base_config["logdir"]),
        "runs": args.runs,
        "episodes_per_run": episodes,
        "llm_model_name": base_config["llm_model_name"],
        "base_seed": base_seed,
        "first_log_number": start_number,
        "started_at": datetime.now().astimezone().isoformat(),
        "tests": [],
    }
    _write_manifest(manifest_path, manifest)

    print(f"Config: {CONFIG_PATH}")
    print(f"Episodes per test (from YAML): {episodes}")
    print(f"Model (from YAML): {base_config['llm_model_name']}")
    print("Sequential log directories:")
    for path in test_dirs:
        print(f"  {path}")

    if str(PROPS_DIR) not in sys.path:
        sys.path.insert(0, str(PROPS_DIR))
    from runner.llm_num_optim_runner import run_training_loop

    previous_dir = Path.cwd()
    try:
        os.chdir(PROPS_DIR)
        for index, test_dir in enumerate(test_dirs):
            number = index + 1
            seed = base_seed + index
            config = deepcopy(base_config)
            config["seed"] = seed
            config["logdir"] = str(test_dir)

            record = {
                "test": number,
                "seed": seed,
                "logdir": str(test_dir),
                "status": "running",
                "started_at": datetime.now().astimezone().isoformat(),
            }
            manifest["tests"].append(record)
            _write_manifest(manifest_path, manifest)
            print("=" * 72, flush=True)
            print(f"Starting test {number}/{args.runs}: seed={seed}", flush=True)
            print(f"Episodes: {episodes}", flush=True)
            print(f"Logs: {test_dir}", flush=True)
            print("=" * 72, flush=True)
            started = time.perf_counter()
            try:
                run_training_loop(**config)
            except BaseException as exc:
                record["status"] = (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                )
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["elapsed_seconds"] = time.perf_counter() - started
                record["finished_at"] = datetime.now().astimezone().isoformat()
                manifest["status"] = record["status"]
                manifest["finished_at"] = record["finished_at"]
                _write_manifest(manifest_path, manifest)
                raise
            record["status"] = "complete"
            record["elapsed_seconds"] = time.perf_counter() - started
            record["finished_at"] = datetime.now().astimezone().isoformat()
            _write_manifest(manifest_path, manifest)

        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        manifest["status"] = "complete"
        _write_manifest(manifest_path, manifest)
    finally:
        os.chdir(previous_dir)

    print(f"All {args.runs} tests completed.")
    print(f"Batch manifest: {manifest_path}")


if __name__ == "__main__":
    main()
