#!/usr/bin/env python3
"""
run_optomerge.py
================
Batch-process every TIFF movie found under ``test_data/`` (recursively) with the
optomerge alignment pipeline and write the aligned RGB output to
``output_temp/``, mirroring the input folder structure.

This is the object-oriented entry point: it drives :class:`optomerge.MergePipeline`
(``RawMovie -> ChannelLayout -> Aligner -> RGBMovie``).

Usage
-----
    python run_optomerge.py [OPTIONS]

Parameters come from three layers, each overriding the previous: built-in
defaults, then any ``--config`` TOML file(s), then explicit CLI flags. The fully
resolved configuration is written to ``<output>/run_config.toml`` for every run,
so a result is reproducible with ``--config <output>/run_config.toml``.

Options
-------
    --config FILE [FILE ...]
                    TOML config file(s), merged left-to-right; CLI flags win.
    --input  DIR    Source directory                       (default: test_data)
    --output DIR    Output directory                        (default: output_temp)
    --suffix STR    Suffix inserted before the extension    (default: _aligned)
    --overwrite     Overwrite existing output files
    --rgb-bits 8|16 RGB output bit depth (default: 8; 8-bit is compact and shown
                    at a fixed 0-255 range, 16-bit auto-contrasts in ImageJ)
    --channel-order STR
                    auto (default) |
                    top_green_fils_bottom_red_heads |
                    top_red_heads_bottom_green_fils
    --frames N      Max frames used for channel/alignment detection (default: all)
    --bg-radius N   Background-subtraction structuring-element radius (default: 10)
    --bunch-size N  Frames per processing block            (default: 100000)
    --use-scrub     Align on upsampled scrub images for sub-pixel accuracy
    --upscale N     Scrub upscaling factor when --use-scrub (default: 4)
    --max-shift PX  Constrain the fitted translation to +/- PX px of the origin,
                    ignoring spurious far-off correlation peaks (0 = off)
    --robust        Use the robust best-of-N calibrator (chunk the movie, score
                    each block, keep acceptance-passing candidates, pick the best)
    --reuse-alignment none|first|FILE
                    Conserve one alignment across each set (see --group-by).
                    none (default): each movie independent.  first: each set's
                    first movie is its reference.  FILE: use that movie as a
                    predetermined reference for every set.  Intensity
                    normalisation is always recomputed per movie.  A movie whose
                    fit fails the acceptance criteria is rejected (not saved).
    --group-by run|token|folder
                    Partition movies into alignment sets (with --reuse-alignment):
                    run (whole batch), token (filename marker), or folder.
    --group-token REGEX
                    Filename marker regex for --group-by token (e.g. "_ch\\d\\d_").
    --verbose       Print per-stage progress inside each file
    --dry-run       List files that would be processed, then exit

A plain-text log is written to ``<output>/run_log.txt`` and the resolved config
to ``<output>/run_config.toml``.

Examples
--------
    python run_optomerge.py
    python run_optomerge.py --config my_run.toml
    python run_optomerge.py --config base.toml --channel-order top_green_fils_bottom_red_heads
    python run_optomerge.py --reuse-alignment first
    python run_optomerge.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Directory containing this script (used to resolve relative input/output paths).
_here = Path(__file__).resolve().parent

try:
    from optomerge import (
        AlignmentNotFoundError,
        Calibration,
        ConservedCalibrator,
        MergePipeline,
        RawMovie,
        Settings,
        group_movies,
    )
except ImportError as e:
    sys.exit(
        "ERROR: Cannot import optomerge.\n"
        "  Install it first (from the repo root):  pip install -e .\n"
        f"  Original error: {e}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHANNEL_ORDERS = (
    "auto",
    "top_green_fils_bottom_red_heads",
    "top_red_heads_bottom_green_fils",
)


# Maps argparse dests to flat Settings field names. Every flag below defaults to
# argparse.SUPPRESS, so only explicitly-passed flags appear in vars(args) and
# thus override the config file (unset flags keep the file/default value).
_CLI_TO_FIELD = {
    "input": "input", "output": "output", "suffix": "suffix", "overwrite": "overwrite",
    "rgb_bits": "rgb_bitdepth",
    "channel_order": "channel_order", "frames": "projection_frames",
    "segmentation": "segmentation",
    "bg_radius": "bg_radius", "bunch_size": "bunch_size",
    "norm_projection": "norm_projection",
    "use_scrub": "use_scrub", "upscale": "upscale", "max_shift": "max_shift",
    "fit_scale_rotation": "fit_scale_rotation", "align_method": "method",
    "deweight_stuck": "deweight_stuck",
    "fit_rotation": "fit_rotation", "fit_scale": "fit_scale",
    "feature_rotation_max_deg": "feature_rotation_max_deg",
    "feature_scale_max_pct": "feature_scale_max_pct",
    "calibration_mode": "mode",
    "group_by": "by", "group_token": "token_pattern",
    "reuse_alignment": "reuse_alignment", "verbose": "verbose", "dry_run": "dry_run",
    "workers": "workers",
}


def _setup_logging(log_path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("optomerge_batch")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


def _collect_tif_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".tif", ".tiff"} and p.is_file()
    )


def _output_path(src_file: Path, src_root: Path, dst_root: Path, suffix: str) -> Path:
    dst = dst_root / src_file.relative_to(src_root)
    return dst.with_name(dst.stem + suffix + ".tif")


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec:02d}s"


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    src: Path,
    dst: Path,
    *,
    calibrator,
    bunch_size: int,
    bg_radius: int,
    projection_frames: int | None,
    verbose: bool,
    logger: logging.Logger,
    shared: "Calibration | None" = None,
    norm_from_max: bool = True,
    norm_exclude: float = 0.0,
    rgb_bitdepth: int = 8,
) -> dict:
    """Calibrate, gate on acceptance, then merge one file.

    ``shared`` is an optional reference :class:`Calibration` to reuse (conserve
    alignment across a set); when given, its layout + transforms are applied and
    only the per-movie intensity limits are recomputed. Otherwise the movie is
    calibrated with ``calibrator`` and rejected if the fit fails acceptance
    (robust calibration raises; single-projection sets ``accepted = False``).
    """
    result: dict = {
        "src": str(src), "dst": str(dst), "success": False, "rejected": False,
        "duration": 0.0, "transforms": None, "shape": None, "score": float("nan"),
        "error": None, "calibration": None,
    }
    t0 = time.perf_counter()
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        active = (ConservedCalibrator(shared, projection_frames=projection_frames,
                                      norm_from_max=norm_from_max, norm_exclude=norm_exclude)
                  if shared is not None else calibrator)
        pipe = MergePipeline(source=src, calibrator=active, bunch_size=bunch_size,
                             bg_radius=bg_radius, projection_frames=projection_frames,
                             verbose=verbose, rgb_bitdepth=rgb_bitdepth)

        movie = RawMovie.open(src)
        try:
            pipe.calibrate(movie)
        except AlignmentNotFoundError as exc:
            result["rejected"] = True
            result["error"] = f"alignment rejected: {exc}"
            logger.warning(f"  [REJECT] {exc}")
            result["duration"] = time.perf_counter() - t0
            return result

        cal = pipe.calibration
        result["calibration"] = cal
        result["score"] = cal.score if cal is not None else float("nan")

        # Gate freshly-fitted (non-conserved) movies on acceptance.
        if shared is None and cal is not None and not cal.accepted:
            result["rejected"] = True
            reason = "; ".join(cal.reasons) or "failed acceptance"
            result["error"] = f"alignment rejected: {reason}"
            logger.warning(f"  [REJECT] {reason}")
            result["duration"] = time.perf_counter() - t0
            return result

        rgb = pipe.merge(movie, output=dst)
        result["shape"] = tuple(rgb.to_array().shape)
        result["transforms"] = {
            name: {
                "t": (round(t.t1, 4), round(t.t2, 4)),
                "rot_deg": round(np.degrees(t.rot), 5),
                "s": (round(t.s1, 5), round(t.s2, 5)),
                "score": round(t.score, 4),
            }
            for name, t in pipe.transforms.items()
        }
        n_ch = len(pipe.resolved_layout.specs) if pipe.resolved_layout else 0
        logger.info(f"  {n_ch} channel(s), score={result['score']:.4f}; "
                    f"transforms: {result['transforms'] or '(reference only)'}")
        logger.info(f"  Saved -> {dst.name}  shape={result['shape']}")
        result["success"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        logger.error(f"  FAILED:\n{result['error']}")
    result["duration"] = time.perf_counter() - t0
    return result


def _calibrate_reference(path: Path, calibrator, logger: logging.Logger) -> "Calibration":
    """Compute a predetermined reference calibration from a movie file."""
    logger.info(f"  Computing predetermined alignment from: {path.name}")
    return calibrator.calibrate(RawMovie.open(path))


# ---------------------------------------------------------------------------
# Parallel (file-level) processing
# ---------------------------------------------------------------------------

class _ListLogger:
    """Minimal logger stand-in that captures records for later replay.

    ``ProcessPoolExecutor`` workers cannot share the parent's file-handler
    logger, so each worker logs into one of these and the parent replays the
    lines (in order, grouped per file) once the worker returns.
    """

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def info(self, msg: str) -> None:
        self.lines.append(("info", msg))

    def warning(self, msg: str) -> None:
        self.lines.append(("warning", msg))

    def error(self, msg: str) -> None:
        self.lines.append(("error", msg))


def _build_pf_kwargs(settings, calibrator, logger) -> dict:
    """Assemble the keyword arguments :func:`process_file` needs from settings."""
    return dict(
        calibrator=calibrator, bunch_size=settings.processing.bunch_size,
        bg_radius=settings.processing.bg_radius,
        projection_frames=settings.channels.projection_frames,
        verbose=settings.runtime.verbose, logger=logger,
        norm_from_max=settings.processing.norm_projection == "max",
        norm_exclude=settings.processing.norm_exclude,
        rgb_bitdepth=settings.io.rgb_bitdepth,
    )


def _cap_worker_threads(thread_cap: int):
    """Bound every layer of threading in a worker to ~``thread_cap`` cores.

    Critical for parallel workers: the kernels have three nested sources of
    threads, and only capping ours leaves the others oversubscribing --
    ``workers`` processes each spawning cv2/BLAS threads across *all* cores
    thrashes (measured: per-movie time 5x worse, no batch speedup). So:

    * cv2 (the warp/morph backend) -> 1 thread per op (``cv2.setNumThreads``);
    * BLAS/OpenMP (numpy/scipy) -> ``thread_cap`` via threadpoolctl if present;
    * our per-frame ``ThreadPoolExecutor`` -> ``thread_cap`` (the frame-level
      parallelism that actually fills this worker's core share).

    Returns a context manager for the BLAS limit (a no-op if threadpoolctl is
    absent), to be held open for the duration of the work.
    """
    import contextlib

    from optomerge import set_default_workers
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:  # pragma: no cover - cv2 optional
        pass
    set_default_workers(thread_cap)
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=thread_cap)
    except Exception:  # pragma: no cover - threadpoolctl optional
        return contextlib.nullcontext()


def _process_one(job: tuple) -> dict:
    """Worker entry point: process one movie in an isolated process.

    ``job`` is ``(settings, src, dst, shared, thread_cap)``, all picklable.
    Caps all threading layers to ``thread_cap`` (see :func:`_cap_worker_threads`)
    so ``workers`` processes don't oversubscribe the cores, rebuilds the
    calibrator from ``settings`` (rather than pickling a live one), and captures
    log output for the parent to replay. Never raises: :func:`process_file`
    already turns per-file failures into a result dict, keeping the batch alive.
    """
    settings, src, dst, shared, thread_cap = job
    buf = _ListLogger()
    calibrator = settings.build_calibrator()
    with _cap_worker_threads(thread_cap):
        result = process_file(Path(src), Path(dst), shared=shared,
                              **_build_pf_kwargs(settings, calibrator, buf))
    result["log"] = buf.lines
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    S = argparse.SUPPRESS
    parser = argparse.ArgumentParser(
        description="Batch-process TIFF movies with optomerge (object-oriented API).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Config file(s): merged left-to-right; explicit CLI flags override them.
    parser.add_argument("--config", nargs="+", default=None, metavar="FILE",
                        help="TOML config file(s); later files and CLI flags override earlier ones")
    # All overriding flags default to SUPPRESS so unset flags don't clobber the
    # config-file / built-in values (see _CLI_TO_FIELD).
    parser.add_argument("--input", default=S, metavar="DIR")
    parser.add_argument("--output", default=S, metavar="DIR")
    parser.add_argument("--suffix", default=S, metavar="STR")
    parser.add_argument("--overwrite", action="store_true", default=S)
    parser.add_argument("--rgb-bits", type=int, default=S, choices=[8, 16], dest="rgb_bits",
                        help="RGB output bit depth (default: 8)")
    parser.add_argument("--channel-order", default=S, choices=_CHANNEL_ORDERS,
                        dest="channel_order", metavar="STR")
    parser.add_argument("--frames", type=int, default=S, metavar="N",
                        help="Max frames for the detection projection")
    parser.add_argument("--segmentation", default=S, choices=["row_profile", "line_search"],
                        metavar="STR", help="channel-finding method (default: row_profile)")
    parser.add_argument("--bg-radius", type=int, default=S, metavar="N", dest="bg_radius")
    parser.add_argument("--norm-projection", default=S, choices=["max", "mean"],
                        dest="norm_projection", metavar="STR",
                        help="projection for normalisation limits (default: max)")
    parser.add_argument("--bunch-size", type=int, default=S, metavar="N", dest="bunch_size")
    parser.add_argument("--use-scrub", action="store_true", default=S, dest="use_scrub")
    parser.add_argument("--upscale", type=int, default=S, metavar="N")
    parser.add_argument("--max-shift", type=float, default=S, metavar="PX", dest="max_shift",
                        help="constrain alignment translation to +/- PX px (0 = unconstrained)")
    parser.add_argument("--no-rotation", action="store_const", const=False, default=S,
                        dest="fit_scale_rotation",
                        help="fit translation only (pin rotation/scale; avoids over-fitting)")
    parser.add_argument("--align-method", default=S, choices=["phase", "feature"],
                        dest="align_method",
                        help="registration algorithm: phase (default) | feature "
                             "(head-to-filament distance, for point-vs-line channels)")
    parser.add_argument("--feature", action="store_const", const="feature", default=S,
                        dest="align_method",
                        help="shorthand for --align-method feature")
    parser.add_argument("--deweight-stuck", action="store_true", default=S,
                        dest="deweight_stuck",
                        help="(feature) down-weight stuck objects (count once, not per-frame)")
    parser.add_argument("--fit-rotation", action="store_true", default=S,
                        dest="fit_rotation",
                        help="(feature) also fit a small bounded rotation between channels")
    parser.add_argument("--fit-scale", action="store_true", default=S,
                        dest="fit_scale",
                        help="(feature) also fit a small bounded isotropic scale between channels")
    parser.add_argument("--rotation-max-deg", type=float, default=S, metavar="DEG",
                        dest="feature_rotation_max_deg",
                        help="(feature) bound on the rotation search (degrees; default 2)")
    parser.add_argument("--scale-max-pct", type=float, default=S, metavar="PCT",
                        dest="feature_scale_max_pct",
                        help="(feature) bound on the scale search (percent; default 2)")
    parser.add_argument("--robust", action="store_const", const="robust", default=S,
                        dest="calibration_mode",
                        help="use the robust best-of-N calibrator")
    parser.add_argument("--reuse-alignment", default=S, dest="reuse_alignment",
                        metavar="none|first|FILE",
                        help="conserve one alignment across each set (see --group-by)")
    parser.add_argument("--group-by", default=S, choices=["run", "token", "folder"],
                        dest="group_by",
                        help="partition movies into alignment sets (with --reuse-alignment)")
    parser.add_argument("--group-token", default=S, dest="group_token", metavar="REGEX",
                        help="filename marker regex for --group-by token")
    parser.add_argument("--workers", type=int, default=S, metavar="N",
                        help="process this many movies in parallel (default 1). Each "
                             "worker caps inner kernel threads to ~cores/N. Not used "
                             "with --reuse-alignment first (kept sequential).")
    parser.add_argument("--verbose", action="store_true", default=S)
    parser.add_argument("--dry-run", action="store_true", default=S, dest="dry_run")
    args = parser.parse_args()

    # Resolution order: built-in defaults -> config file(s) -> explicit CLI flags.
    try:
        settings = Settings.from_toml(*args.config) if args.config else Settings()
    except (OSError, KeyError, RuntimeError) as exc:
        sys.exit(f"ERROR: could not load config: {exc}")
    overrides = {_CLI_TO_FIELD[dest]: val for dest, val in vars(args).items()
                 if dest in _CLI_TO_FIELD}
    settings = settings.with_overrides(**overrides)

    src_root = (_here / settings.io.input).resolve()
    dst_root = (_here / settings.io.output).resolve()
    suffix = settings.io.suffix
    overwrite = settings.io.overwrite
    if not src_root.is_dir():
        sys.exit(f"ERROR: Input directory not found: {src_root}")

    files = _collect_tif_files(src_root)
    if not files:
        sys.exit(f"No .tif files found under {src_root}")

    if settings.runtime.dry_run:
        print(f"Dry run — {len(files)} file(s) found under {src_root}:\n")
        for f in files:
            out = _output_path(f, src_root, dst_root, suffix)
            print(f"  {f.relative_to(src_root)}\n    -> {out.relative_to(dst_root)}\n")
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    log_path = dst_root / "run_log.txt"
    logger = _setup_logging(log_path, settings.runtime.verbose)

    # Provenance: record the fully-resolved config next to the output so the run
    # is reproducible (loads back via Settings.from_toml).
    config_path = dst_root / "run_config.toml"
    config_path.write_text(settings.to_toml(), encoding="utf-8")

    calibrator = settings.build_calibrator()
    reuse = settings.runtime.reuse_alignment
    groups = group_movies(files, settings.grouping.by, settings.grouping.token_pattern)

    logger.info("=" * 70)
    logger.info("optomerge batch processor (OOP)")
    logger.info(f"  Input  : {src_root}")
    logger.info(f"  Output : {dst_root}")
    logger.info(f"  Files  : {len(files)}  in {len(groups)} set(s)")
    logger.info(f"  Config : {', '.join(args.config) if args.config else '(built-in defaults)'}")
    logger.info(f"  Channel order : {settings.channels.channel_order}")
    logger.info(f"  Calibrator    : {settings.calibration.mode}"
                + (f"  chunk={settings.calibration.chunk_size} "
                   f"min_cands={settings.calibration.min_candidates} "
                   f"max_trials={settings.calibration.max_trials}"
                   if settings.calibration.mode == "robust" else ""))
    logger.info(f"  BG radius     : {settings.processing.bg_radius}")
    logger.info(f"  Bunch size    : {settings.processing.bunch_size}")
    logger.info(f"  Workers       : {max(1, settings.runtime.workers)}"
                + ("  (parallel across files)" if settings.runtime.workers > 1 else ""))
    logger.info(f"  Conserve      : {reuse}"
                + (f"  (group-by {settings.grouping.by})" if reuse != "none" else ""))
    if settings.channels.projection_frames:
        logger.info(f"  Projection frames: {settings.channels.projection_frames}")
    logger.info(f"  Resolved config saved to: {config_path.name}")
    logger.info("=" * 70)

    # Predetermined alignment from a reference FILE (applied to every set).
    predetermined: "Calibration | None" = None
    if reuse not in ("none", "first"):
        cand = Path(reuse)
        if not cand.is_absolute():
            cand = (_here / reuse).resolve()
        if not cand.exists():
            matches = [f for f in files if f.name == cand.name]
            if not matches:
                sys.exit(f"ERROR: reuse_alignment reference not found: {reuse}")
            cand = matches[0]
        try:
            predetermined = _calibrate_reference(cand, calibrator, logger)
        except AlignmentNotFoundError as exc:
            sys.exit(f"ERROR: predetermined reference did not pass acceptance: {exc}")
        if not predetermined.accepted:
            sys.exit("ERROR: predetermined reference failed acceptance: "
                     + "; ".join(predetermined.reasons))
        logger.info(f"  Predetermined transform(s): "
                    + ", ".join(f"{n}: rot={np.degrees(t.rot):.3f}deg "
                                f"t=({t.t1:.2f},{t.t2:.2f})"
                                for n, t in predetermined.transforms.items()))

    workers = max(1, settings.runtime.workers)
    # File-level parallelism is only safe when files are independent: --reuse none
    # (each movie stands alone) or a predetermined reference FILE (shared, already
    # computed). --reuse first has a within-set dependency (the first movie seeds
    # the rest), so it stays sequential.
    parallel = workers > 1 and reuse != "first"
    if workers > 1 and reuse == "first":
        logger.info("  Note: --workers is ignored with --reuse-alignment first "
                    "(each set's first movie must seed the rest; kept sequential).")

    results: list[dict] = []
    idx = 0
    t_batch = time.perf_counter()

    if parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        thread_cap = max(1, (os.cpu_count() or 1) // workers)

        jobs = []  # (idx, src, dst) for files that actually need processing
        for gkey, members in groups:
            for src in members:
                idx += 1
                dst = _output_path(src, src_root, dst_root, suffix)
                if dst.exists() and not overwrite:
                    logger.info(f"\n[{idx}/{len(files)}] {src.relative_to(src_root)}")
                    logger.info("  [SKIP] output exists (use --overwrite)")
                    results.append({"src": str(src), "success": False, "skipped": True,
                                    "rejected": False, "error": None, "duration": 0.0})
                    continue
                jobs.append((idx, src, dst))

        logger.info(f"\nProcessing {len(jobs)} file(s) across {workers} worker "
                    f"process(es), ~{thread_cap} inner thread(s) each ...")
        # shared = predetermined reference (None when reuse == "none").
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fut_meta = {
                pool.submit(_process_one,
                            (settings, str(src), str(dst), predetermined, thread_cap)):
                    (jidx, src)
                for (jidx, src, dst) in jobs
            }
            for fut in as_completed(fut_meta):
                jidx, src = fut_meta[fut]
                r = fut.result()
                r["index"] = jidx
                logger.info(f"\n[{jidx}/{len(files)}] {src.relative_to(src_root)}")
                for level, msg in r.pop("log", []):
                    getattr(logger, level)(msg)
                status = "OK" if r["success"] else ("REJECT" if r.get("rejected") else "FAILED")
                logger.info(f"  [{status}] {_fmt_seconds(r['duration'])}")
                results.append(r)
    else:
        for gkey, members in groups:
            set_shared = predetermined  # None unless reuse == FILE
            label = "" if settings.grouping.by == "run" else f" [{gkey}]"
            for src in members:
                idx += 1
                dst = _output_path(src, src_root, dst_root, suffix)
                logger.info(f"\n[{idx}/{len(files)}]{label} {src.relative_to(src_root)}")
                if dst.exists() and not overwrite:
                    logger.info("  [SKIP] output exists (use --overwrite)")
                    results.append({"src": str(src), "success": False, "skipped": True,
                                    "rejected": False, "error": None, "duration": 0.0})
                    continue

                pf_kwargs = _build_pf_kwargs(settings, calibrator, logger)
                if reuse == "none":
                    r = process_file(src, dst, shared=None, **pf_kwargs)
                elif set_shared is None:
                    # First movie of the set defines its alignment.
                    r = process_file(src, dst, shared=None, **pf_kwargs)
                    if r["success"] and r.get("calibration") is not None:
                        set_shared = r["calibration"]
                    elif r.get("rejected"):
                        logger.warning("  set reference rejected; remaining movies in this "
                                       "set will be aligned independently")
                else:
                    r = process_file(src, dst, shared=set_shared, **pf_kwargs)

                r["index"] = idx
                results.append(r)
                status = "OK" if r["success"] else ("REJECT" if r.get("rejected") else "FAILED")
                logger.info(f"  [{status}] {_fmt_seconds(r['duration'])}")

    # ---- Summary ----
    total = time.perf_counter() - t_batch
    n_ok = sum(1 for r in results if r["success"])
    n_rejected = sum(1 for r in results if r.get("rejected"))
    n_skipped = sum(1 for r in results if r.get("skipped"))
    n_failed = sum(1 for r in results
                   if not r["success"] and r.get("error") and not r.get("rejected"))
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info(f"  Files     : {len(results)}")
    logger.info(f"  Succeeded : {n_ok}")
    logger.info(f"  Rejected  : {n_rejected}  (failed acceptance)")
    logger.info(f"  Skipped   : {n_skipped}  (output existed)")
    logger.info(f"  Failed    : {n_failed}")
    logger.info(f"  Total time: {_fmt_seconds(total)}")
    if n_rejected:
        logger.info("\nRejected files:")
        for r in results:
            if r.get("rejected"):
                logger.info(f"  - {r['src']}: {r.get('error')}")
    if n_failed:
        logger.info("\nFailed files:")
        for r in results:
            if not r["success"] and r.get("error") and not r.get("rejected"):
                logger.info(f"  - {r['src']}")
    logger.info(f"\nLog written to: {log_path}")
    logger.info("=" * 70)
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
