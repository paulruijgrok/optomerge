#!/usr/bin/env python3
"""
Batch processing framework template.

Adapt this for any pipeline that needs to run unattended across many datasets,
each with its own config. Implements the standard pattern from
pipeline-engineering-standards:

  - per-dataset config discovery (base config + per-dataset overrides)
  - fail-isolated execution: one dataset's failure never aborts the batch
  - structured per-dataset logging + a batch-level summary at the end
  - resume-by-default: already-completed datasets are skipped on re-run
  - no interactive prompts: anything unresolved fails that dataset, not the batch

Usage:
    python batch_runner_template.py --configs-dir configs/ --output-dir results/
    python batch_runner_template.py --configs-dir configs/ --output-dir results/ --force
"""

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # config loading falls back to JSON if PyYAML isn't installed


# --------------------------------------------------------------------------
# Replace this with your pipeline's actual entry point. It should take a
# fully-resolved config dict and an output directory, and either complete
# successfully or raise — don't swallow errors here, let the runner handle it.
# --------------------------------------------------------------------------
def run_pipeline(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """
    Run the analysis pipeline for a single dataset.

    Return a small dict of summary metrics to record alongside the result
    (e.g. {"n_frames": 500, "n_tracks": 12}) — kept in the batch summary
    for a quick eyeball of results without opening every output directory.
    """
    raise NotImplementedError("Replace run_pipeline() with the actual analysis call.")


@dataclass
class DatasetResult:
    name: str
    status: str  # "success" | "failed" | "skipped"
    duration_s: float = 0.0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def load_config(base_config_path: Path | None, dataset_config_path: Path) -> dict[str, Any]:
    """Merge a base config (if given) with a per-dataset override file."""

    def _load(path: Path) -> dict[str, Any]:
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("PyYAML not installed; install it or use JSON configs.")
            return yaml.safe_load(text) or {}
        return json.loads(text)

    config: dict[str, Any] = {}
    if base_config_path is not None:
        config.update(_load(base_config_path))
    config.update(_load(dataset_config_path))
    return config


def discover_dataset_configs(configs_dir: Path) -> list[Path]:
    """Each file directly in configs_dir (excluding a base config) is one dataset."""
    patterns = ("*.yaml", "*.yml", "*.json")
    configs = []
    for pattern in patterns:
        configs.extend(configs_dir.glob(pattern))
    return sorted(p for p in configs if p.stem != "base")


def already_completed(output_dir: Path) -> bool:
    """A dataset is considered done if it has a SUCCESS marker from a prior run."""
    return (output_dir / "_SUCCESS").exists()


def setup_dataset_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"batch.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def run_batch(configs_dir: Path, output_root: Path, force: bool = False) -> list[DatasetResult]:
    base_config_path = configs_dir / "base.yaml"
    if not base_config_path.exists():
        base_config_path = configs_dir / "base.json"
    if not base_config_path.exists():
        base_config_path = None

    dataset_configs = discover_dataset_configs(configs_dir)
    if not dataset_configs:
        print(f"No dataset configs found in {configs_dir}", file=sys.stderr)
        return []

    output_root.mkdir(parents=True, exist_ok=True)
    results: list[DatasetResult] = []

    for config_path in dataset_configs:
        name = config_path.stem
        dataset_output_dir = output_root / name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_dataset_logger(name, dataset_output_dir / "run.log")

        if not force and already_completed(dataset_output_dir):
            logger.info("Already completed in a previous run; skipping (use --force to redo).")
            results.append(DatasetResult(name=name, status="skipped"))
            print(f"[skip]    {name} (already completed)")
            continue

        # Clear any stale marker before attempting, so a crash mid-run never
        # leaves a dataset looking "done" on the next resume.
        success_marker = dataset_output_dir / "_SUCCESS"
        success_marker.unlink(missing_ok=True)

        start = time.monotonic()
        try:
            config = load_config(base_config_path, config_path)
            logger.info("Starting run with config: %s", config)
            metrics = run_pipeline(config, dataset_output_dir)
            duration = time.monotonic() - start
            success_marker.write_text("")
            logger.info("Completed successfully in %.1fs", duration)
            results.append(DatasetResult(name=name, status="success", duration_s=duration, metrics=metrics))
            print(f"[ok]      {name} ({duration:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - intentional: isolate failures per dataset
            duration = time.monotonic() - start
            err_text = "".join(traceback.format_exception(exc))
            logger.error("Failed after %.1fs:\n%s", duration, err_text)
            results.append(DatasetResult(name=name, status="failed", duration_s=duration, error=str(exc)))
            print(f"[FAILED]  {name}: {exc}")
            # Deliberately no re-raise: one dataset's failure must not abort the batch.

    return results


def write_summary(results: list[DatasetResult], output_root: Path) -> None:
    summary = {
        "total": len(results),
        "succeeded": sum(r.status == "success" for r in results),
        "failed": sum(r.status == "failed" for r in results),
        "skipped": sum(r.status == "skipped" for r in results),
        "datasets": [
            {
                "name": r.name,
                "status": r.status,
                "duration_s": round(r.duration_s, 2),
                "error": r.error,
                "metrics": r.metrics,
            }
            for r in results
        ],
    }
    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n--- Batch summary ---")
    print(f"  total:     {summary['total']}")
    print(f"  succeeded: {summary['succeeded']}")
    print(f"  failed:    {summary['failed']}")
    print(f"  skipped:   {summary['skipped']}")
    if summary["failed"]:
        print("  failed datasets:")
        for d in summary["datasets"]:
            if d["status"] == "failed":
                print(f"    - {d['name']}: {d['error']}")
    print(f"  full summary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, required=True,
                         help="Directory containing one config file per dataset "
                              "(plus an optional base.yaml/base.json).")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Root directory for per-dataset outputs and the batch summary.")
    parser.add_argument("--force", action="store_true",
                         help="Re-run datasets even if already marked complete.")
    args = parser.parse_args()

    results = run_batch(args.configs_dir, args.output_dir, force=args.force)
    write_summary(results, args.output_dir)

    if any(r.status == "failed" for r in results):
        sys.exit(1)  # non-zero exit so CI / cron / overnight runs surface the failure


if __name__ == "__main__":
    main()
