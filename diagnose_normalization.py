#!/usr/bin/env python3
"""
diagnose_normalization.py
=========================
Show *why* a channel saturates: for each channel it prints the normalisation
limits (vmin/vmax) under several strategies and the resulting per-frame
saturation fraction (fraction of pixels that clip to 1.0), so we can see whether
vmax is set too low.

Strategies compared for vmax:
  mean            - max of the mean-projection crop (old behaviour)
  max_raw         - max of the max-projection crop (no robustness)
  max_despeckle   - max of the 3x3-median max-projection crop (current default)
  max_pct99.9     - 99.9th percentile of the max-projection crop
  temporal_p99.5  - per-pixel 99.5th percentile over time, then max (cosmic-ray
                    robust *and* keeps sharp heads)

Usage
-----
    python diagnose_normalization.py --file "movie.tif" \\
        --channel-order top_green_fils_bottom_red_heads --frames 200 --sample 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_pkg = _here / "optomerge"
if _pkg.is_dir() and str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from optomerge import ChannelLayout, RawMovie
from optomerge._kernels.processing import crop_channel

_CHANNEL_ORDERS = ("auto", "top_green_fils_bottom_red_heads", "top_red_heads_bottom_green_fils")


def _despeckle(a):
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(a, float), size=3)


def diagnose(src: Path, channel_order: str, frames, sample: int):
    movie = RawMovie.open(src)
    H, W, N = movie.shape
    max_proj = movie.max_projection(frames)
    mean_proj = movie.mean_projection(frames)
    layout = ChannelLayout(name=channel_order, channel_order=channel_order).resolve(
        max_proj, mean_proj)

    # Sample frames evenly for per-frame statistics + temporal percentile.
    idx = np.linspace(0, N - 1, min(sample, N)).astype(int)
    data = movie.to_array()          # (H, W, N)
    stack = data[:, :, idx]          # sampled frames

    print(f"\n{src.name}")
    print(f"  {N} frames, {H}x{W};  channel_order={channel_order}\n")

    for spec in layout.specs:
        name, b, m = spec.name, spec.bounds, spec.mask
        maxc = crop_channel(max_proj, b, m)
        meanc = crop_channel(mean_proj, b, m)
        # per-frame channel crops (sampled)
        framec = np.stack([crop_channel(stack[:, :, k], b, m) for k in range(stack.shape[2])], axis=2)

        strategies = {
            "mean":           float(meanc.max()),
            "max_raw":        float(maxc.max()),
            "max_despeckle":  float(_despeckle(maxc).max()),
            "max_pct99.9":    float(np.percentile(maxc, 99.9)),
            "temporal_p99.5": float(np.percentile(framec, 99.5, axis=2).max()),
        }
        vmin = float(maxc.min())
        # typical per-frame peak: median over sampled frames of each frame's max
        per_frame_peak = np.median([framec[:, :, k].max() for k in range(framec.shape[2])])

        print(f"  [{name}]  vmin={vmin:.1f}  median per-frame peak={per_frame_peak:.1f}")
        for label, vmax in strategies.items():
            rng = max(vmax - vmin, 1e-9)
            sat = float(np.mean((framec - vmin) / rng >= 1.0))   # fraction clipped
            flag = "  <-- saturates" if sat > 0.001 else ""
            print(f"      vmax[{label:<14}] = {vmax:9.1f}   saturated px = {sat*100:6.3f}%{flag}")
        print()


def main():
    ap = argparse.ArgumentParser(description="Diagnose channel normalisation / saturation.")
    ap.add_argument("--file", required=True)
    ap.add_argument("--channel-order", default="auto", choices=_CHANNEL_ORDERS, dest="channel_order")
    ap.add_argument("--frames", type=int, default=None, help="Frames for projections")
    ap.add_argument("--sample", type=int, default=60, help="Frames sampled for per-frame stats")
    args = ap.parse_args()
    src = Path(args.file)
    if not src.exists():
        sys.exit(f"File not found: {src}")
    diagnose(src, args.channel_order, args.frames, args.sample)


if __name__ == "__main__":
    main()
