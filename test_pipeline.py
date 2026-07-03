#!/usr/bin/env python3
"""
test_pipeline.py
================
A step-by-step diagnostic for the optomerge alignment pipeline. It drives the
object-oriented API (:mod:`optomerge`) one stage at a time — load, project,
resolve channel layout, fit the alignment transform, assemble the aligned RGB
movie — and saves a diagnostic PNG after each stage so the result of every step
can be inspected by eye.

Usage
-----
    python test_pipeline.py                      # scan test_data/
    python test_pipeline.py --file "movie 01.tif"
    python test_pipeline.py --no-diag --frames 200
    python test_pipeline.py --dry-run

Requirements
------------
    pip install -e optomerge/      # numpy, scipy, scikit-image, tifffile
    pip install matplotlib         # for the diagnostic images
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Locate the optomerge package ──────────────────────────────────────────
_here = Path(__file__).resolve().parent
_pkg = _here / "optomerge"
if _pkg.is_dir() and str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

try:
    from optomerge import (
        ChannelLayout,
        PhaseCorrelationAligner,
        RawMovie,
        RGBMovie,
    )
    from optomerge._kernels.transform import zero_pad_images
except ImportError as exc:
    sys.exit(
        f"Cannot import optomerge ({exc}).\n"
        f"Install it with:  pip install -e optomerge/\n"
        f"or ensure the optomerge/ package folder is next to this script."
    )

_CHANNEL_ORDERS = (
    "auto",
    "top_green_fils_bottom_red_heads",
    "top_red_heads_bottom_green_fils",
)


# ─────────────────────────────────────────────────────────────────────────────
# Stage runner — drives the OOP pipeline and returns intermediate artifacts
# ─────────────────────────────────────────────────────────────────────────────

class StageResult:
    """Bundle of intermediate artifacts produced while stepping the pipeline."""

    def __init__(self) -> None:
        self.data: Optional[np.ndarray] = None         # (H, W, N) raw
        self.max_proj: Optional[np.ndarray] = None
        self.mean_proj: Optional[np.ndarray] = None
        self.layout: Optional[ChannelLayout] = None
        self.limits: dict = {}                          # name -> (vmin, vmax)
        self.transforms: dict = {}                      # name -> Transform
        self.rgb: Optional[np.ndarray] = None           # (H, W, 3, N)


def run_stages(
    src: Path,
    channel_order: str,
    projection_frames: Optional[int],
    bg_radius: int,
    use_scrub: bool,
    upscale: int,
    save_diag: bool,
    out_dir: Path,
    stem: str,
    verbose: bool,
) -> StageResult:
    """Execute the pipeline stage by stage, saving a diagnostic after each."""
    sr = StageResult()

    # ── 1. Load ───────────────────────────────────────────────────────────
    print("  [1/5] Loading ...")
    movie = RawMovie.open(src)
    sr.data = movie.to_array()                       # (H, W, N)

    # ── 2. Project ────────────────────────────────────────────────────────
    print("  [2/5] Projecting ...")
    sr.max_proj = movie.max_projection(projection_frames)
    sr.mean_proj = movie.mean_projection(projection_frames)
    if save_diag:
        save_diag_projection(sr, out_dir, stem)

    # ── 3. Resolve layout (channel finding) ───────────────────────────────
    print("  [3/5] Finding channels and computing alignment ...")
    candidate = ChannelLayout(name=channel_order, channel_order=channel_order)
    sr.layout = candidate.resolve(sr.max_proj, sr.mean_proj, verbose=verbose)

    # ── 4. Calibrate + fit alignment transforms ───────────────────────────
    proj_channels = {c.name: c.calibrate() for c in sr.layout.split(sr.mean_proj)}
    sr.limits = {n: (c.vmin, c.vmax) for n, c in proj_channels.items()}
    reference = next(c for c in proj_channels.values() if c.reference)
    aligner = PhaseCorrelationAligner(use_scrub=use_scrub, upscale=upscale)
    for spec in sr.layout.moving_specs:
        sr.transforms[spec.name] = aligner.align(reference, proj_channels[spec.name])

    for name, t in sr.transforms.items():
        print(f"         {name}: t=({t.t1:.3f},{t.t2:.3f})  "
              f"rot={np.degrees(t.rot):.4f}°  s=({t.s1:.5f},{t.s2:.5f})  "
              f"score={t.score:.4f}")

    if save_diag:
        save_diag_segmentation(sr, out_dir, stem)
        save_diag_channels(sr, out_dir, stem)
        save_diag_alignment(sr, out_dir, stem)

    # ── 5. Assemble aligned RGB movie ─────────────────────────────────────
    print("  [4/5] Aligning frames ...")
    channels = sr.layout.split(sr.data)
    for ch in channels:
        ch.vmin, ch.vmax = sr.limits[ch.name]
        if ch.name in sr.transforms:
            ch.transform = sr.transforms[ch.name]
    rgb_movie = RGBMovie.from_channels(channels, sr.transforms, bg_radius=bg_radius)
    sr.rgb = rgb_movie.to_array()
    if save_diag:
        save_diag_rgb_frame(sr, out_dir, stem)

    # ── 6. Save ───────────────────────────────────────────────────────────
    print("  [5/5] Saving ...")
    out_tif = out_dir / f"{stem}_aligned.tif"
    rgb_movie.save(out_tif)
    print(f"  Output folder: {out_dir}")
    return sr


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic figures
# ─────────────────────────────────────────────────────────────────────────────

def _disp_im(ax, im, title, cmap="hot", vmin=None, vmax=None):
    ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_diag_projection(sr: StageResult, out_dir: Path, stem: str):
    """Diag 01 – max-intensity projection."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    _disp_im(ax, sr.max_proj, f"Max projection\n{stem}", cmap="gray")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diag_01_projection.png", dpi=150)
    plt.close(fig)


def save_diag_segmentation(sr: StageResult, out_dir: Path, stem: str):
    """Diag 02 – detected channel boundaries overlaid on the projection."""
    proj = sr.max_proj
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _disp_im(axes[0], proj, "Max projection", cmap="gray")

    vmin, vmax = np.percentile(proj, [1, 99])
    axes[1].imshow(proj, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    axes[1].set_title("Detected channels", fontsize=9)
    axes[1].axis("off")

    colours = {"green": "lime", "red": "red"}
    for spec in sr.layout.specs:
        b = spec.bounds
        if b is None:
            continue
        r0, r1 = b[0, 0], b[0, 1]
        c0, c1 = b[1, 0], b[1, 1]
        rect = mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0, linewidth=2,
            edgecolor=colours.get(spec.color, "yellow"), facecolor="none",
            label=f"{spec.name} ({spec.color})",
        )
        axes[1].add_patch(rect)
    axes[1].legend(loc="lower right", fontsize=7, framealpha=0.7)
    fig.suptitle(stem, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diag_02_segmentation.png", dpi=150)
    plt.close(fig)


def save_diag_channels(sr: StageResult, out_dir: Path, stem: str):
    """Diag 03 – cropped channels from frame 0."""
    frame0 = sr.data[:, :, 0]
    channels = sr.layout.split(frame0)
    cmaps = {"green": "Greens_r", "red": "Reds_r"}

    if len(channels) == 1:
        ch = channels[0]
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        _disp_im(ax, ch.data, f"{ch.name} (frame 0)", cmap=cmaps.get(ch.color, "hot"))
    else:
        fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4))
        for ax, ch in zip(np.atleast_1d(axes), channels):
            _disp_im(ax, ch.data, f"{ch.name} ({ch.color}, frame 0)",
                     cmap=cmaps.get(ch.color, "hot"))
    fig.suptitle(stem, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diag_03_channels.png", dpi=150)
    plt.close(fig)


def save_diag_alignment(sr: StageResult, out_dir: Path, stem: str):
    """Diag 04 – before/after alignment overlay using mean projections."""
    if not sr.transforms:
        return  # single-channel: nothing to align

    proj_channels = {c.name: c for c in sr.layout.split(sr.mean_proj)}
    reference = next(c for c in proj_channels.values() if c.reference)
    moving = next(c for c in proj_channels.values() if not c.reference)
    transform = sr.transforms[moving.name]

    reference.vmin, reference.vmax = sr.limits[reference.name]
    moving.vmin, moving.vmax = sr.limits[moving.name]
    g_norm = reference.normalized()
    r_norm = moving.normalized()
    r_pad, g_pad = zero_pad_images(r_norm, g_norm)

    H, W = r_pad.shape
    before = np.zeros((H, W, 3))
    before[:, :, 0] = r_pad
    before[:, :, 1] = g_pad

    r_aligned = transform.apply(r_pad)
    after = np.zeros((H, W, 3))
    after[:, :, 0] = r_aligned
    after[:, :, 1] = g_pad

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.clip(before, 0, 1), aspect="auto")
    axes[0].set_title(f"Before alignment\n(red={moving.name}, green={reference.name})", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(np.clip(after, 0, 1), aspect="auto")
    axes[1].set_title(
        f"After alignment\nt=({transform.t1:.2f},{transform.t2:.2f})  "
        f"rot={np.degrees(transform.rot):.3f}°  s=({transform.s1:.4f},{transform.s2:.4f})",
        fontsize=9,
    )
    axes[1].axis("off")
    fig.suptitle(stem, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diag_04_alignment.png", dpi=150)
    plt.close(fig)


def save_diag_rgb_frame(sr: StageResult, out_dir: Path, stem: str):
    """Diag 05 – first frame of the aligned RGB output."""
    if sr.rgb is None:
        return
    frame0 = np.clip(sr.rgb[:, :, :, 0], 0, 1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(frame0, aspect="auto")
    ax.set_title(f"Aligned RGB – frame 0\n{stem}", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diag_05_rgb_frame.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_file(
    src: Path,
    dst_dir: Path,
    channel_order: str,
    projection_frames: Optional[int],
    bg_radius: int,
    use_scrub: bool,
    upscale: int,
    save_diag: bool,
    verbose: bool,
) -> dict:
    """Run the full pipeline on *src*, saving outputs + diagnostics to *dst_dir*."""
    result = dict(src=str(src), success=False, duration=0.0,
                  alignment=None, n_frames=None, error=None)
    t0 = time.perf_counter()
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        stem = src.stem
        sr = run_stages(
            src, channel_order, projection_frames, bg_radius,
            use_scrub, upscale, save_diag, dst_dir, stem, verbose,
        )
        result["n_frames"] = sr.data.shape[2]
        if sr.transforms:
            t = next(iter(sr.transforms.values()))
            result["alignment"] = {
                "t1": round(t.t1, 4), "t2": round(t.t2, 4),
                "rot_deg": round(np.degrees(t.rot), 5),
                "s1": round(t.s1, 5), "s2": round(t.s2, 5),
                "score": round(t.score, 4),
            }
        result["success"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        print(f"  FAILED:\n{result['error']}")
    result["duration"] = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Step-by-step diagnostic test for the optomerge pipeline (OOP API).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", default="test_data", metavar="DIR",
                   help="Root directory of raw movies (default: test_data)")
    p.add_argument("--output", default="output_temp", metavar="DIR",
                   help="Root directory for output (default: output_temp)")
    p.add_argument("--file", default=None, metavar="FILE",
                   help="Process a single file instead of scanning --input")
    p.add_argument("--channel-order", default="auto", choices=_CHANNEL_ORDERS,
                   dest="channel_order", metavar="STR", help="Channel order (default: auto)")
    p.add_argument("--frames", type=int, default=None, metavar="N",
                   help="Max frames for projection (default: all)")
    p.add_argument("--bg-radius", type=int, default=10, metavar="N", dest="bg_radius",
                   help="Background subtraction radius (default: 10)")
    p.add_argument("--use-scrub", action="store_true", dest="use_scrub",
                   help="Align on upsampled scrub images for sub-pixel accuracy")
    p.add_argument("--upscale", type=int, default=4, metavar="N",
                   help="Scrub upscaling factor when --use-scrub (default: 4)")
    p.add_argument("--no-diag", action="store_true",
                   help="Skip saving diagnostic PNG images")
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed per-step progress")
    p.add_argument("--dry-run", action="store_true",
                   help="List files that would be processed, then exit")
    args = p.parse_args()

    dst_root = (_here / args.output).resolve()

    if args.file:
        files = [Path(args.file).resolve()]
        src_root = files[0].parent
    else:
        src_root = (_here / args.input).resolve()
        if not src_root.is_dir():
            sys.exit(f"Input directory not found: {src_root}")
        files = sorted(
            f for f in src_root.rglob("*")
            if f.suffix.lower() in {".tif", ".tiff"} and f.is_file()
        )
        if not files:
            sys.exit(f"No .tif files found under {src_root}")

    if args.dry_run:
        print(f"Dry run — {len(files)} file(s) found:\n")
        for f in files:
            try:
                rel = f.relative_to(src_root)
            except ValueError:
                rel = Path(f.name)
            dst = dst_root / rel.parent / rel.stem
            print(f"  {rel}\n    -> {dst.relative_to(dst_root.parent)}/\n")
        return

    print("=" * 68)
    print("optomerge diagnostic test pipeline (OOP)")
    print(f"  Input : {src_root}")
    print(f"  Output: {dst_root}")
    print(f"  Files : {len(files)}")
    print(f"  Channel order: {args.channel_order}   BG radius: {args.bg_radius}")
    print(f"  Aligner: phase-correlation" + (f" (scrub x{args.upscale})" if args.use_scrub else ""))
    print(f"  Diag images: {'no' if args.no_diag else 'yes'}")
    print("=" * 68)

    results = []
    t_total = time.perf_counter()
    for idx, src in enumerate(files, 1):
        try:
            rel = src.relative_to(src_root)
        except ValueError:
            rel = Path(src.name)
        dst_dir = dst_root / rel.parent / rel.stem
        print(f"\n[{idx}/{len(files)}] {rel}")
        result = process_file(
            src=src, dst_dir=dst_dir,
            channel_order=args.channel_order, projection_frames=args.frames,
            bg_radius=args.bg_radius, use_scrub=args.use_scrub, upscale=args.upscale,
            save_diag=not args.no_diag, verbose=args.verbose,
        )
        results.append(result)
        mins, secs = divmod(int(result["duration"]), 60)
        print(f"  [{'OK' if result['success'] else 'FAILED'}] {mins}m {secs:02d}s")

    elapsed = time.perf_counter() - t_total
    n_ok = sum(1 for r in results if r["success"])
    n_fail = len(results) - n_ok
    print("\n" + "=" * 68)
    print(f"DONE  {n_ok}/{len(results)} succeeded  "
          f"({int(elapsed // 60)}m {int(elapsed % 60):02d}s total)")
    if n_fail:
        print("\nFailed files:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['src']}")
    if n_ok:
        print("\nAlignment summary:")
        for r in results:
            if r["success"] and r["alignment"]:
                a = r["alignment"]
                name = Path(r["src"]).name
                print(f"  {name[:50]:<50}  t=({a['t1']:.2f},{a['t2']:.2f})  "
                      f"rot={a['rot_deg']:.3f}°  score={a['score']:.3f}")
    print("=" * 68)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
