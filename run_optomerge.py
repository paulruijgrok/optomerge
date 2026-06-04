#!/usr/bin/env python3
"""
run_optomerge.py
================
Batch-process all TIFF movie files found inside ``test_data/`` (and every
sub-directory) with the optomerge alignment pipeline, and write the aligned
RGB output to ``output_temp/``, preserving the original folder structure.

Usage
-----
    python run_optomerge.py [OPTIONS]

Options
-------
    --input  DIR    Source directory  (default: test_data)
    --output DIR    Output directory  (default: output_temp)
    --suffix STR    Suffix appended to each output filename before the
                    extension  (default: _aligned)
    --channel-order STR
                    One of:  auto  (default)
                             top_green_fils_bottom_red_heads
                             top_red_heads_bottom_green_fils
    --frames N      Maximum number of frames to use for the channel/alignment
                    detection projection  (default: all)
    --bg-radius N   Radius of the background-subtraction structuring element
                    (default: 10)
    --bg-method STR
                    Background subtraction method: morph_open (default, MATLAB
                    equivalent) or gaussian (10–30× faster, similar quality)
    --backend STR   Computational backend: auto (default, uses cv2 when available),
                    cv2 (force OpenCV), or scipy (force scipy/skimage)
    --alignment-mode STR
                    Strategy for computing the channel alignment:
                      robust        (default) project multiple frame chunks,
                                    score each, pick the best — matches MATLAB
                                    findChannelsAndAlignment
                      single        one projection of all (or --frames) frames
    --robust-chunk-size N
                    Frames per projection chunk  (default: 400)
    --robust-min-candidates N
                    Min valid candidates before selecting best  (default: 10)
    --robust-max-trials N
                    Max chunks tried before giving up  (default: 30)
    --align-method STR
                    Frame-alignment method applied after computing alignment
                    parameters:
                      auto          (default) use fft_shift when rotation < 0.5°
                                    and scale deviation < 0.5%; otherwise affine
                      affine        full affine transform (most accurate, slowest)
                      fft_shift     sub-pixel Fourier shift (2–3× faster than
                                    affine; ignores residual rotation/scale)
                      quick_and_dirty  integer-pixel shift, no interpolation
                                    (~20–50× faster, ±0.5 px accuracy)
    --reuse-alignment none|first|FILE
                    Reuse channel bounds and alignment transform across the
                    batch.  none (default): each movie is processed
                    independently.  first: the first movie becomes the
                    reference.  FILE: use the specified file as the reference.
                    Intensity normalisation is always computed per-movie.
    --no-progress   Suppress the live progress bars (useful when piping output)
    --workers N     Number of CPU threads for parallel frame processing.
                    Default: all logical CPUs.  Use 1 to disable parallelism.
    --parallel-files
                    Process multiple movie files simultaneously (one per core).
                    When active, frame-level workers is set to 1 to avoid
                    nested parallelism.  Best for many short movies.
    --verbose       Print per-frame progress inside each file
    --dry-run       List files that would be processed without doing any work

The script also writes a plain-text log file ``output_temp/run_log.txt``
summarising every file processed, its detected alignment parameters, and
any errors encountered.

Examples
--------
    # Basic run with defaults (all CPUs, morph_open background)
    python run_optomerge.py

    # Faster background subtraction using Gaussian blur
    python run_optomerge.py --bg-method gaussian

    # Use 4 threads for frame processing
    python run_optomerge.py --workers 4

    # Process multiple files in parallel (good for many short movies)
    python run_optomerge.py --parallel-files

    # Force a specific channel order and use only the first 200 frames for
    # alignment detection
    python run_optomerge.py --channel-order top_green_fils_bottom_red_heads \\
                            --frames 200

    # Preview what would be processed
    python run_optomerge.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locate the optomerge package (works even before pip-install if the
# package folder sits next to this script).
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
_pkg  = _here / "optomerge"
if _pkg.is_dir() and str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

try:
    from optomerge import AlignmentPipeline, SharedAlignment
    from optomerge.config import OptomergeConfig
    from optomerge.io import load_frames, save_rgb_tiff
except ImportError as e:
    sys.exit(
        f"ERROR: Cannot import optomerge.\n"
        f"  Make sure the package is installed ('pip install -e optomerge/')\n"
        f"  or that the optomerge/ folder is next to this script.\n"
        f"  Original error: {e}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path, verbose: bool) -> logging.Logger:
    """Return a logger that writes to both stdout and *log_path*."""
    logger = logging.getLogger("optomerge_batch")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")

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
    """Recursively collect .tif / .tiff files, sorted for reproducibility."""
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".tif", ".tiff"} and p.is_file()
    )


def _mirror_path(src_file: Path, src_root: Path, dst_root: Path) -> Path:
    """Translate *src_file* under *src_root* to the equivalent under *dst_root*."""
    rel = src_file.relative_to(src_root)
    return dst_root / rel


def _output_path(src_file: Path, src_root: Path, dst_root: Path,
                 suffix: str) -> Path:
    """Build the output file path, inserting *suffix* before the extension."""
    dst = _mirror_path(src_file, src_root, dst_root)
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
    channel_order: str,
    projection_frames: int | None,
    bg_radius: int,
    bg_method: str,
    n_workers: int | None,
    backend: str,
    alignment_mode: str,
    robust_chunk_size: int,
    robust_min_candidates: int,
    robust_max_trials: int,
    align_method: str,
    show_progress: bool,
    verbose: bool,
    logger: logging.Logger | None = None,
    shared_alignment: "SharedAlignment | None" = None,
    extract_shared: bool = False,
) -> dict:
    """Run the full optomerge pipeline on a single file.

    Returns a result dictionary with keys:
        success, duration, alignment, n_frames, shape, error,
        shared_alignment (only when extract_shared=True and success=True)

    ``logger`` may be None when called from a subprocess worker (the worker
    sets up its own logging after forking).
    ``shared_alignment``: if provided, channel bounds and alignment transform
        are reused from this reference instead of being recomputed.
    ``extract_shared``: if True, include a ``shared_alignment`` key in the
        result so the caller can use it as a reference for subsequent files.
    """
    if logger is None:
        logger = logging.getLogger("optomerge_worker")
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.setLevel(logging.INFO)

    result: dict = {
        "src": str(src),
        "dst": str(dst),
        "success": False,
        "duration": 0.0,
        "alignment": None,
        "n_frames": None,
        "shape": None,
        "error": None,
        "shared_alignment": None,
    }

    t0 = time.perf_counter()
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        if shared_alignment is not None:
            logger.info(
                f"  Reusing alignment from reference: "
                f"{shared_alignment.reference_file}"
            )

        pipe = AlignmentPipeline(
            filename=src,
            channel_order=channel_order,
            projection_frames=projection_frames,
            bg_radius=bg_radius,
            bg_method=bg_method,
            n_workers=n_workers,
            backend=backend,
            alignment_mode=alignment_mode,
            robust_chunk_size=robust_chunk_size,
            robust_min_candidates=robust_min_candidates,
            robust_max_trials=robust_max_trials,
            align_method=align_method,
            show_progress=show_progress,
            verbose=verbose,
            shared_alignment=shared_alignment,
        )

        # Load and project
        pipe.load()
        H, W, N = pipe._movie.shape
        result["n_frames"] = N
        logger.info(f"  Loaded {N} frames, {H}×{W} px")

        pipe.project()
        pipe.find_channels()

        if pipe.bounds2 is not None:
            logger.info(
                f"  Two channels: ch1 rows {pipe.bounds1[0,0]}–{pipe.bounds1[0,1]}, "
                f"cols {pipe.bounds1[1,0]}–{pipe.bounds1[1,1]}"
            )
        else:
            logger.info("  Single channel detected")

        pipe.compute_alignment()

        if pipe.alignment is not None:
            a = pipe.alignment
            result["alignment"] = {
                "t1": round(a.t1, 4), "t2": round(a.t2, 4),
                "rot_deg": round(np.degrees(a.rot), 5),
                "s1": round(a.s1, 5), "s2": round(a.s2, 5),
                "score": round(a.score, 4),
            }
            logger.info(
                f"  Alignment — t=({a.t1:.3f}, {a.t2:.3f}), "
                f"rot={np.degrees(a.rot):.4f}°, "
                f"s=({a.s1:.4f}, {a.s2:.4f}), score={a.score:.4f}"
            )

        rgb = pipe.align_frames()
        result["shape"] = tuple(rgb.shape)

        # Save
        save_rgb_tiff(dst, rgb)
        logger.info(f"  Saved  →  {dst.relative_to(dst.parent.parent.parent)}")

        # Optionally extract SharedAlignment for batch reuse
        if extract_shared and pipe.alignment is not None:
            result["shared_alignment"] = pipe.get_shared_alignment()

        result["success"] = True

    except Exception:
        err = traceback.format_exc()
        result["error"] = err
        logger.error(f"  FAILED:\n{err}")

    result["duration"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process TIFF movies with optomerge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ---- Config file (loaded first; all other flags override it) ----
    parser.add_argument("--config", default=None, metavar="FILE",
                        help=(
                            "Path to a TOML config file.  If not given, the script "
                            "looks for optomerge.toml in the current directory, then "
                            "~/.config/optomerge/optomerge.toml.  All CLI flags "
                            "override values from the config file."
                        ))

    # ---- I/O ----
    # All defaults are None so we can distinguish "not supplied" from "supplied"
    # when merging with the config.
    parser.add_argument("--input",  default=None, metavar="DIR",
                        help="Root directory of raw movies  [cfg: io.input]")
    parser.add_argument("--output", default=None, metavar="DIR",
                        help="Root directory for output  [cfg: io.output]")
    parser.add_argument("--suffix", default=None, metavar="STR",
                        help="Suffix added to output filenames  [cfg: io.suffix]")
    parser.add_argument("--overwrite", action="store_true", default=None,
                        help="Overwrite existing output files  [cfg: io.overwrite]")

    # ---- Projection ----
    parser.add_argument("--channel-order", default=None,
                        choices=["auto",
                                 "top_green_fils_bottom_red_heads",
                                 "top_red_heads_bottom_green_fils"],
                        metavar="STR",
                        help="Channel order  [cfg: projection.channel_order]")
    parser.add_argument("--frames", type=int, default=None, metavar="N",
                        help="Max frames for projection  [cfg: projection.frames]")
    parser.add_argument("--bg-radius", type=int, default=None, metavar="N",
                        help="Background-subtraction radius  [cfg: projection.bg_radius]")
    parser.add_argument("--bg-method", default=None,
                        choices=["morph_open", "gaussian"],
                        metavar="STR",
                        help="Background method  [cfg: projection.bg_method]")

    # ---- Alignment ----
    parser.add_argument("--alignment-mode", default=None,
                        choices=["robust", "single"],
                        dest="alignment_mode",
                        metavar="STR",
                        help="Alignment strategy  [cfg: alignment.mode]")
    parser.add_argument("--robust-chunk-size", type=int, default=None, metavar="N",
                        dest="robust_chunk_size",
                        help="Frames per chunk for robust alignment  [cfg: alignment.robust_chunk_size]")
    parser.add_argument("--robust-min-candidates", type=int, default=None, metavar="N",
                        dest="robust_min_candidates",
                        help="Min valid candidates for robust alignment  [cfg: alignment.robust_min_candidates]")
    parser.add_argument("--robust-max-trials", type=int, default=None, metavar="N",
                        dest="robust_max_trials",
                        help="Max chunks tried in robust alignment  [cfg: alignment.robust_max_trials]")
    parser.add_argument("--reuse-alignment", default=None,
                        dest="reuse_alignment",
                        metavar="none|first|FILE",
                        help=(
                            "Reuse channel bounds and alignment transform across "
                            "the batch.  'none': independent per movie.  'first': "
                            "first file is the reference.  FILE: specific reference "
                            "file.  [cfg: alignment.reuse]"
                        ))

    # ---- Frames ----
    parser.add_argument("--align-method", default=None,
                        choices=["auto", "affine", "fft_shift", "quick_and_dirty"],
                        dest="align_method",
                        metavar="STR",
                        help="Frame-alignment method  [cfg: frames.align_method]")

    # ---- Compute ----
    parser.add_argument("--backend", default=None,
                        choices=["auto", "cv2", "scipy"],
                        metavar="STR",
                        help="Computational backend  [cfg: compute.backend]")
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="CPU threads per movie  [cfg: compute.n_workers]")
    parser.add_argument("--parallel-files", action="store_true", default=None,
                        help="Process multiple files in parallel  [cfg: compute.parallel_files]")

    # ---- Logging ----
    parser.add_argument("--no-progress", action="store_true", default=None,
                        help="Suppress live progress bars  [cfg: logging.show_progress = false]")
    parser.add_argument("--verbose", action="store_true", default=None,
                        help="Per-frame progress messages  [cfg: logging.verbose]")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="List files without processing  [cfg: logging.dry_run]")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config, then apply CLI overrides
    # Resolution order: defaults → config file → CLI flags
    # ------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent

    cfg = OptomergeConfig.find_and_load(
        explicit_path=Path(args.config) if args.config else None,
        search_cwd=True,
    )

    # Apply CLI overrides — only when the flag was explicitly provided
    if args.input          is not None: cfg.io.input              = args.input
    if args.output         is not None: cfg.io.output             = args.output
    if args.suffix         is not None: cfg.io.suffix             = args.suffix
    if args.overwrite:                  cfg.io.overwrite          = True

    if args.channel_order  is not None: cfg.projection.channel_order = args.channel_order
    if args.frames         is not None: cfg.projection.frames     = args.frames
    if args.bg_radius      is not None: cfg.projection.bg_radius  = args.bg_radius
    if args.bg_method      is not None: cfg.projection.bg_method  = args.bg_method

    if args.alignment_mode      is not None: cfg.alignment.mode                  = args.alignment_mode
    if args.robust_chunk_size   is not None: cfg.alignment.robust_chunk_size     = args.robust_chunk_size
    if args.robust_min_candidates is not None: cfg.alignment.robust_min_candidates = args.robust_min_candidates
    if args.robust_max_trials   is not None: cfg.alignment.robust_max_trials     = args.robust_max_trials
    if args.reuse_alignment     is not None: cfg.alignment.reuse                 = args.reuse_alignment

    if args.align_method   is not None: cfg.frames.align_method   = args.align_method

    if args.backend        is not None: cfg.compute.backend       = args.backend
    if args.workers        is not None: cfg.compute.n_workers     = args.workers
    if args.parallel_files:             cfg.compute.parallel_files = True

    if args.no_progress:                cfg.logging.show_progress = False
    if args.verbose:                    cfg.logging.verbose       = True
    if args.dry_run:                    cfg.logging.dry_run       = True

    # Convenience aliases used throughout the rest of this function
    src_root = (script_dir / cfg.io.input).resolve()
    dst_root = (script_dir / cfg.io.output).resolve()

    if not src_root.is_dir():
        sys.exit(f"ERROR: Input directory not found: {src_root}")

    # Collect files
    files = _collect_tif_files(src_root)
    if not files:
        sys.exit(f"No .tif files found under {src_root}")

    # ---- Dry run ----
    if cfg.logging.dry_run:
        print(f"Dry run — {len(files)} file(s) found under {src_root}:\n")
        cfg_src = cfg.source_path()
        if cfg_src:
            print(f"  Config: {cfg_src}\n")
        for f in files:
            out = _output_path(f, src_root, dst_root, cfg.io.suffix)
            print(f"  {f.relative_to(src_root)}")
            print(f"    → {out.relative_to(dst_root)}\n")
        return

    # ---- Real run ----
    dst_root.mkdir(parents=True, exist_ok=True)
    log_path = dst_root / "run_log.txt"
    logger = _setup_logging(log_path, cfg.logging.verbose)

    cfg_src = cfg.source_path()
    logger.info("=" * 70)
    logger.info("optomerge batch processor")
    logger.info(f"  Config : {cfg_src if cfg_src else '(built-in defaults)'}")
    logger.info(f"  Input  : {src_root}")
    logger.info(f"  Output : {dst_root}")
    logger.info(f"  Files  : {len(files)}")
    logger.info(f"  Channel order: {cfg.projection.channel_order}")
    logger.info(
        f"  BG radius: {cfg.projection.bg_radius}  "
        f"method: {cfg.projection.bg_method}  "
        f"backend: {cfg.compute.backend}"
    )
    logger.info(
        f"  Alignment mode: {cfg.alignment.mode}"
        + (
            f"  chunk={cfg.alignment.robust_chunk_size}  "
            f"min_cands={cfg.alignment.robust_min_candidates}  "
            f"max_trials={cfg.alignment.robust_max_trials}"
            if cfg.alignment.mode == "robust" else ""
        )
    )
    logger.info(f"  Align method: {cfg.frames.align_method}")
    logger.info(f"  Reuse alignment: {cfg.alignment.reuse}")
    if cfg.projection.frames:
        logger.info(f"  Projection frames: {cfg.projection.frames}")

    n_workers_frame = cfg.compute.n_workers
    if cfg.compute.parallel_files:
        n_workers_frame = 1
        n_file_workers = multiprocessing.cpu_count()
        logger.info(f"  File-level parallelism: {n_file_workers} processes")
    else:
        n_file_workers = 1
        if n_workers_frame is None:
            logger.info("  Frame-level parallelism: all CPUs (threads)")
        else:
            logger.info(f"  Frame-level parallelism: {n_workers_frame} threads")
    logger.info("=" * 70)

    results = []
    t_batch = time.perf_counter()

    show_progress = cfg.logging.show_progress
    frame_progress = show_progress and (n_file_workers == 1)

    # ------------------------------------------------------------------
    # Shared-alignment setup
    # ------------------------------------------------------------------
    reuse = cfg.alignment.reuse
    shared_align: "SharedAlignment | None" = None
    ref_file: "Path | None" = None

    if reuse != "none":
        if reuse == "first":
            ref_file = files[0]
        else:
            candidate = Path(reuse)
            if not candidate.is_absolute():
                candidate = (script_dir / reuse).resolve()
            if not candidate.exists():
                matches = [f for f in files if f.name == candidate.name]
                if matches:
                    candidate = matches[0]
                else:
                    sys.exit(
                        f"ERROR: --reuse-alignment reference file not found: {reuse}"
                    )
            ref_file = candidate

        if n_file_workers > 1:
            logger.warning(
                "  --reuse-alignment is ignored in --parallel-files mode "
                "(processes cannot share state). Use sequential mode."
            )
            ref_file = None
        else:
            logger.info(
                f"  Reference file for shared alignment: "
                f"{ref_file.relative_to(src_root)}"
            )

    # ------------------------------------------------------------------
    # Helper: common keyword args for process_file
    # ------------------------------------------------------------------
    def _common_kwargs(show_prog: bool = frame_progress) -> dict:
        return dict(
            channel_order=cfg.projection.channel_order,
            projection_frames=cfg.projection.frames,
            bg_radius=cfg.projection.bg_radius,
            bg_method=cfg.projection.bg_method,
            n_workers=n_workers_frame,
            backend=cfg.compute.backend,
            alignment_mode=cfg.alignment.mode,
            robust_chunk_size=cfg.alignment.robust_chunk_size,
            robust_min_candidates=cfg.alignment.robust_min_candidates,
            robust_max_trials=cfg.alignment.robust_max_trials,
            align_method=cfg.frames.align_method,
            show_progress=show_prog,
            verbose=cfg.logging.verbose,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Sequential path (possibly with shared alignment)
    # ------------------------------------------------------------------
    if n_file_workers == 1:
        # Process reference file first (if not already first in list)
        if ref_file is not None and ref_file != files[0]:
            logger.info(f"\n[ref] {ref_file.relative_to(src_root)}")
            ref_dst = _output_path(ref_file, src_root, dst_root, cfg.io.suffix)
            ref_result = process_file(
                src=ref_file,
                dst=ref_dst,
                extract_shared=True,
                **_common_kwargs(),
            )
            ref_result["index"] = 0
            results.append(ref_result)
            if ref_result["success"] and ref_result["shared_alignment"] is not None:
                shared_align = ref_result["shared_alignment"]
                logger.info(
                    f"  SharedAlignment extracted: "
                    f"rot={shared_align.alignment.rot:.5f}  "
                    f"t=({shared_align.alignment.t1:.3f}, {shared_align.alignment.t2:.3f})"
                )
            else:
                logger.warning(
                    "  Reference file failed or produced no alignment — "
                    "falling back to independent alignment for all files."
                )

        for idx, src in enumerate(files, 1):
            dst = _output_path(src, src_root, dst_root, cfg.io.suffix)
            logger.info(f"\n[{idx}/{len(files)}] {src.relative_to(src_root)}")

            is_ref = (src == ref_file)
            result = process_file(
                src=src,
                dst=dst,
                shared_alignment=None if is_ref else shared_align,
                extract_shared=(is_ref and ref_file == files[0]),
                **_common_kwargs(),
            )
            result["index"] = idx
            results.append(result)

            # Capture shared alignment from the first file if ref == first
            if (
                is_ref
                and ref_file == files[0]
                and result["success"]
                and result["shared_alignment"] is not None
                and shared_align is None
            ):
                shared_align = result["shared_alignment"]
                logger.info(
                    f"  SharedAlignment extracted: "
                    f"rot={shared_align.alignment.rot:.5f}  "
                    f"t=({shared_align.alignment.t1:.3f}, {shared_align.alignment.t2:.3f})"
                )

            status = "OK" if result["success"] else "FAILED"
            logger.info(f"  [{status}] {_fmt_seconds(result['duration'])}")

    # ------------------------------------------------------------------
    # Parallel path (no shared alignment support)
    # ------------------------------------------------------------------
    else:
        job_args = [
            (
                src,
                _output_path(src, src_root, dst_root, cfg.io.suffix),
                cfg.projection.channel_order,
                cfg.projection.frames,
                cfg.projection.bg_radius,
                cfg.projection.bg_method,
                n_workers_frame,
                cfg.compute.backend,
                cfg.alignment.mode,
                cfg.alignment.robust_chunk_size,
                cfg.alignment.robust_min_candidates,
                cfg.alignment.robust_max_trials,
                cfg.frames.align_method,
                frame_progress,
                cfg.logging.verbose,
                None,   # logger — worker creates its own
            )
            for src in files
        ]

        def _worker(job):
            src, dst, co, pf, bgr, bgm, nw, bk, almode, rcs, rmc, rmt, am, sp, vb, _ = job
            return process_file(src, dst, co, pf, bgr, bgm, nw, bk,
                                almode, rcs, rmc, rmt, am, sp, vb)

        with ProcessPoolExecutor(max_workers=n_file_workers) as pool:
            file_results = list(pool.map(_worker, job_args))

        for idx, (src, result) in enumerate(zip(files, file_results), 1):
            result["index"] = idx
            results.append(result)
            status = "OK" if result["success"] else "FAILED"
            logger.info(
                f"[{idx}/{len(files)}] {src.relative_to(src_root)}"
                f"  [{status}] {_fmt_seconds(result['duration'])}"
            )

    # ---- Summary ----
    total_time = time.perf_counter() - t_batch
    n_ok     = sum(1 for r in results if r["success"])
    n_failed = len(results) - n_ok

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info(f"  Total files : {len(results)}")
    logger.info(f"  Succeeded   : {n_ok}")
    logger.info(f"  Failed      : {n_failed}")
    logger.info(f"  Total time  : {_fmt_seconds(total_time)}")

    if n_failed:
        logger.info("\nFailed files:")
        for r in results:
            if not r["success"]:
                logger.info(f"  • {r['src']}")

    logger.info(f"\nLog written to: {log_path}")
    logger.info("=" * 70)

    # Exit with non-zero code if any file failed (useful for CI / scripting)
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
