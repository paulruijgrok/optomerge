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
    --channel-order STR
                    auto (default) |
                    top_green_fils_bottom_red_heads |
                    top_red_heads_bottom_green_fils
    --frames N      Max frames used for channel/alignment detection (default: all)
    --bg-radius N   Background-subtraction structuring-element radius (default: 10)
    --bunch-size N  Frames per processing block            (default: 100000)
    --use-scrub     Align on upsampled scrub images for sub-pixel accuracy
    --upscale N     Scrub upscaling factor when --use-scrub (default: 4)
    --reuse-alignment none|first|FILE
                    Reuse channel layout + alignment transform across the batch.
                    none (default): each movie independent.  first: first movie
                    is the reference.  FILE: use that file as the reference.
                    Intensity normalisation is always recomputed per movie.
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
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locate the optomerge package (works before pip-install if the package folder
# sits next to this script: ./optomerge/optomerge/).
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
_pkg = _here / "optomerge"
if _pkg.is_dir() and str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

try:
    from optomerge import ChannelLayout, MergePipeline, PhaseCorrelationAligner, Settings
except ImportError as e:
    sys.exit(
        "ERROR: Cannot import optomerge.\n"
        "  Install it with 'pip install -e optomerge/' or make sure the\n"
        "  optomerge/ package folder sits next to this script.\n"
        f"  Original error: {e}"
    )


# ---------------------------------------------------------------------------
# Shared-alignment pipeline: reuse a reference layout + transforms, but
# recompute per-movie intensity limits (matches the original SharedAlignment).
# ---------------------------------------------------------------------------

class _SharedPipeline(MergePipeline):
    def __init__(self, *args, shared_layout, shared_transforms, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_layout = shared_layout
        self._shared_transforms = shared_transforms

    def calibrate(self, movie):  # type: ignore[override]
        self.resolved_layout = self._shared_layout
        mean_proj = movie.mean_projection(self.projection_frames)
        proj_channels = {c.name: c.calibrate() for c in self.resolved_layout.split(mean_proj)}
        self._limits = {name: (c.vmin, c.vmax) for name, c in proj_channels.items()}
        self.transforms = dict(self._shared_transforms)
        self._log(f"Reusing shared layout ({self.resolved_layout.name}) + transforms")


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
    "channel_order": "channel_order", "frames": "projection_frames",
    "bg_radius": "bg_radius", "bunch_size": "bunch_size",
    "use_scrub": "use_scrub", "upscale": "upscale",
    "reuse_alignment": "reuse_alignment", "verbose": "verbose", "dry_run": "dry_run",
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
    layout: ChannelLayout,
    aligner: PhaseCorrelationAligner,
    bunch_size: int,
    bg_radius: int,
    projection_frames: int | None,
    verbose: bool,
    logger: logging.Logger,
    shared: tuple | None = None,
    extract_shared: bool = False,
) -> dict:
    """Run the OOP pipeline on a single file.

    ``shared`` is an optional ``(resolved_layout, transforms)`` pair to reuse.
    When ``extract_shared`` is True the result includes the pipeline's resolved
    layout + transforms so the caller can reuse them for later files.
    """
    result: dict = {
        "src": str(src), "dst": str(dst), "success": False,
        "duration": 0.0, "transforms": None, "shape": None,
        "error": None, "shared": None,
    }
    t0 = time.perf_counter()
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        common = dict(
            source=src, layout=layout, aligner=aligner, bunch_size=bunch_size,
            bg_radius=bg_radius, projection_frames=projection_frames, verbose=verbose,
        )
        if shared is not None:
            pipe = _SharedPipeline(shared_layout=shared[0], shared_transforms=shared[1], **common)
        else:
            pipe = MergePipeline(**common)

        rgb = pipe.run(output=dst)

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
        logger.info(f"  {n_ch} channel(s); transforms: {result['transforms'] or '(reference only)'}")
        logger.info(f"  Saved -> {dst.name}  shape={result['shape']}")

        if extract_shared and pipe.resolved_layout is not None:
            result["shared"] = (pipe.resolved_layout, dict(pipe.transforms))
        result["success"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        logger.error(f"  FAILED:\n{result['error']}")
    result["duration"] = time.perf_counter() - t0
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
    parser.add_argument("--channel-order", default=S, choices=_CHANNEL_ORDERS,
                        dest="channel_order", metavar="STR")
    parser.add_argument("--frames", type=int, default=S, metavar="N",
                        help="Max frames for the detection projection")
    parser.add_argument("--bg-radius", type=int, default=S, metavar="N", dest="bg_radius")
    parser.add_argument("--bunch-size", type=int, default=S, metavar="N", dest="bunch_size")
    parser.add_argument("--use-scrub", action="store_true", default=S, dest="use_scrub")
    parser.add_argument("--upscale", type=int, default=S, metavar="N")
    parser.add_argument("--reuse-alignment", default=S, dest="reuse_alignment",
                        metavar="none|first|FILE")
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

    layout = settings.build_layout()
    aligner = settings.build_aligner()
    reuse = settings.runtime.reuse_alignment

    logger.info("=" * 70)
    logger.info("optomerge batch processor (OOP)")
    logger.info(f"  Input  : {src_root}")
    logger.info(f"  Output : {dst_root}")
    logger.info(f"  Files  : {len(files)}")
    logger.info(f"  Config : {', '.join(args.config) if args.config else '(built-in defaults)'}")
    logger.info(f"  Channel order : {settings.channels.channel_order}")
    logger.info(f"  BG radius     : {settings.processing.bg_radius}   "
                f"bunch size: {settings.processing.bunch_size}")
    logger.info(f"  Aligner       : phase-correlation"
                + (f" (scrub x{settings.alignment.upscale})" if settings.alignment.use_scrub else ""))
    logger.info(f"  Reuse         : {reuse}")
    if settings.channels.projection_frames:
        logger.info(f"  Projection frames: {settings.channels.projection_frames}")
    logger.info(f"  Resolved config saved to: {config_path.name}")
    logger.info("=" * 70)

    # ---- Resolve the reference file for shared alignment ----
    ref_file: Path | None = None
    if reuse != "none":
        if reuse == "first":
            ref_file = files[0]
        else:
            cand = Path(reuse)
            if not cand.is_absolute():
                cand = (_here / reuse).resolve()
            if not cand.exists():
                matches = [f for f in files if f.name == cand.name]
                if not matches:
                    sys.exit(f"ERROR: reuse_alignment reference not found: {reuse}")
                cand = matches[0]
            ref_file = cand
        logger.info(f"  Reference for shared alignment: {ref_file.relative_to(src_root)}")

    def _common_kwargs():
        return dict(
            layout=layout, aligner=aligner, bunch_size=settings.processing.bunch_size,
            bg_radius=settings.processing.bg_radius,
            projection_frames=settings.channels.projection_frames,
            verbose=settings.runtime.verbose, logger=logger,
        )

    results: list[dict] = []
    shared: tuple | None = None
    t_batch = time.perf_counter()

    # Process the reference first (if not already the first file) so its layout
    # + transforms are available to the rest of the batch.
    if ref_file is not None and ref_file != files[0]:
        logger.info(f"\n[ref] {ref_file.relative_to(src_root)}")
        ref_dst = _output_path(ref_file, src_root, dst_root, suffix)
        if ref_dst.exists() and not overwrite:
            logger.warning("  Reference output exists; reprocessing to extract shared alignment.")
        r = process_file(ref_file, ref_dst, extract_shared=True, **_common_kwargs())
        r["index"] = 0
        results.append(r)
        if r["success"] and r["shared"] is not None:
            shared = r["shared"]
        else:
            logger.warning("  Reference failed — falling back to independent alignment.")

    for idx, src in enumerate(files, 1):
        dst = _output_path(src, src_root, dst_root, suffix)
        logger.info(f"\n[{idx}/{len(files)}] {src.relative_to(src_root)}")
        if dst.exists() and not overwrite:
            logger.info("  [SKIP] output exists (use --overwrite)")
            continue
        is_ref = (src == ref_file)
        r = process_file(
            src, dst,
            shared=None if is_ref else shared,
            extract_shared=(is_ref and ref_file == files[0]),
            **_common_kwargs(),
        )
        r["index"] = idx
        results.append(r)
        if is_ref and ref_file == files[0] and r["success"] and r["shared"] is not None and shared is None:
            shared = r["shared"]
        logger.info(f"  [{'OK' if r['success'] else 'FAILED'}] {_fmt_seconds(r['duration'])}")

    # ---- Summary ----
    total = time.perf_counter() - t_batch
    n_ok = sum(1 for r in results if r["success"])
    n_failed = sum(1 for r in results if not r["success"] and r.get("error"))
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info(f"  Processed : {len(results)}")
    logger.info(f"  Succeeded : {n_ok}")
    logger.info(f"  Failed    : {n_failed}")
    logger.info(f"  Total time: {_fmt_seconds(total)}")
    if n_failed:
        logger.info("\nFailed files:")
        for r in results:
            if not r["success"] and r.get("error"):
                logger.info(f"  - {r['src']}")
    logger.info(f"\nLog written to: {log_path}")
    logger.info("=" * 70)
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
