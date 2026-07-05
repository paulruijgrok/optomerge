#!/usr/bin/env python3
"""
diagnose_segmentation.py
========================
Visualise the internal stages of channel segmentation on one movie, to see
*why* the detected channel crops land where they do.

It renders a multi-panel figure:

  1. Max-intensity projection (what segmentation sees).
  2. k-means "channel pixel" mask (bright / channel pixels).
  3. "boundary pixel" mask (background / edge pixels the line-search runs on).
  4. Row-intensity profile — mean intensity per row, and the fraction of
     channel-pixels per row. This is the signal the (planned) row-profile
     gap-detection method uses: two humps (the two channels) separated by a
     valley (the gap between the OptoSplit halves). The detected split and the
     final channel row-bounds are overlaid.
  5. The projection with the detected channel boxes overlaid.

Usage
-----
    python diagnose_segmentation.py --file "movie.tif" \\
        --channel-order top_green_fils_bottom_red_heads --frames 200

Requires matplotlib (``pip install -e "optomerge/[dev]"``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_here = Path(__file__).resolve().parent
_pkg = _here / "optomerge"
if _pkg.is_dir() and str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

try:
    from optomerge import RawMovie
    from optomerge._kernels.segmentation import (
        _gen_boundary_segmentation_image,
        find_channel_bounds,
    )
except ImportError as exc:
    sys.exit(f"Cannot import optomerge ({exc}). Install with: pip install -e 'optomerge/[dev]'")

_CHANNEL_ORDERS = (
    "auto",
    "top_green_fils_bottom_red_heads",
    "top_red_heads_bottom_green_fils",
)


def _row_profile(image: np.ndarray) -> np.ndarray:
    """Mean intensity per row, normalised to [0, 1]."""
    prof = image.mean(axis=1)
    rng = prof.max() - prof.min()
    return (prof - prof.min()) / rng if rng > 0 else prof * 0.0


def diagnose(src: Path, channel_order: str, max_frames: int | None, out_path: Path):
    movie = RawMovie.open(src)
    max_proj = movie.max_projection(max_frames)
    mean_proj = movie.mean_projection(max_frames)
    H, W = max_proj.shape

    # Internal segmentation stages.
    channel_mask, boundary_mask = _gen_boundary_segmentation_image(
        mean_proj, "quick_kmeans", num_clusters=10, cluster_level=4
    )

    # Row signals.
    intensity_prof = _row_profile(max_proj)
    channel_frac = channel_mask.mean(axis=1)  # fraction of channel pixels per row

    # Final detected bounds.
    b1, s1, b2, s2 = find_channel_bounds(max_proj, mean_projection=mean_proj,
                                         channel_order=channel_order, verbose=False)
    boxes = [("ch1", b1)] + ([("ch2", b2)] if b2 is not None else [])

    # ---- Figure ----
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(max_proj, cmap="gray", aspect="auto")
    axes[0].set_title("1. Max projection")

    axes[1].imshow(channel_mask, cmap="viridis", aspect="auto")
    axes[1].set_title("2. k-means channel mask")

    axes[2].imshow(boundary_mask, cmap="viridis", aspect="auto")
    axes[2].set_title("3. boundary mask (line-search input)")

    # Row profile panel: plot signals vs row index (y = row, to line up with images).
    rows = np.arange(H)
    axes[3].plot(intensity_prof, rows, label="mean intensity", color="k")
    axes[3].plot(channel_frac, rows, label="channel-pixel frac", color="tab:green")
    axes[3].axhline(H // 2, color="gray", ls=":", label="H/2 (current split)")
    for name, b in boxes:
        axes[3].axhline(b[0, 0], color="tab:red", ls="--", lw=0.8)
        axes[3].axhline(b[0, 1], color="tab:red", ls="--", lw=0.8)
    axes[3].invert_yaxis()
    axes[3].set_title("4. row profile")
    axes[3].set_xlabel("normalised signal")
    axes[3].set_ylabel("row")
    axes[3].legend(fontsize=7, loc="lower right")

    vmin, vmax = np.percentile(max_proj, [1, 99])
    axes[4].imshow(max_proj, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    colours = {"ch1": "lime", "ch2": "red"}
    for name, b in boxes:
        r0, r1 = b[0, 0], b[0, 1]
        c0, c1 = b[1, 0], b[1, 1]
        axes[4].add_patch(mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0, linewidth=2,
            edgecolor=colours[name], facecolor="none", label=name))
    axes[4].set_title("5. detected boxes")
    axes[4].legend(fontsize=7, loc="lower right")

    for ax in [axes[0], axes[1], axes[2], axes[4]]:
        ax.axis("off")

    n_ch = 1 + (b2 is not None)
    fig.suptitle(f"{src.name}   |   channel_order={channel_order}   |   {n_ch} channel(s)",
                 fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"channels detected: {n_ch}")
    for name, b in boxes:
        print(f"  {name}: rows {b[0,0]}-{b[0,1]}  cols {b[1,0]}-{b[1,1]}")
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Diagnose channel segmentation on one movie.")
    ap.add_argument("--file", required=True, help="Path to a TIFF movie")
    ap.add_argument("--channel-order", default="auto", choices=_CHANNEL_ORDERS,
                    dest="channel_order")
    ap.add_argument("--frames", type=int, default=None, help="Max frames for projection")
    ap.add_argument("--output", default=None, help="Output PNG path")
    args = ap.parse_args()

    src = Path(args.file)
    if not src.exists():
        sys.exit(f"File not found: {src}")
    out = Path(args.output) if args.output else (
        _here / "output_temp" / f"{src.stem}_segdiag.png")
    diagnose(src, args.channel_order, args.frames, out)


if __name__ == "__main__":
    main()
